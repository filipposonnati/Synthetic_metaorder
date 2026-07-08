import os
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import datetime

# Impostiamo il device (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo in uso: {device}")

# =====================================================================
# 1. DEFINIZIONE DELL'ARCHITETTURA DELLA RETE
# =====================================================================

class LOBGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128, num_layers=2):
        super(LOBGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM per catturare la dipendenza temporale
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)

        # Strati lineari separati per gestire le diverse scale e attivazioni
        self.fc_time_price_vol = nn.Linear(hidden_dim, output_dim - 1)
        self.fc_sign = nn.Linear(hidden_dim, 1)

        # Sigmoide per costringere la probabilità del segno tra 0 e 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # z ha dimensione: [batch_size, sequence_length, latent_dim]
        out, _ = self.lstm(z)

        # Generiamo tempo, prezzo e volume (valori continui lineari)
        cont_features = self.fc_time_price_vol(out)

        # Generiamo la probabilità del segno
        sign_prob = self.sigmoid(self.fc_sign(out))

        # Concateniamo tutto per ottenere l'output finale: [batch_size, sequence_length, 4]
        generated_data = torch.cat([cont_features, sign_prob], dim=-1)
        return generated_data


class LOBDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(LOBDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x rappresenta la sequenza di trade (reale o sintetica)
        out, _ = self.lstm(x)
        # Prendiamo solo l'output dell'ultimo step temporale per la classificazione
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# =====================================================================
# 2. CARICAMENTO E POOLING DI TUTTI I GIORNI
# =====================================================================
#
# Invece di trattare ogni giorno come un mini-dataset a sé (che lascia sia
# il discriminatore che il generatore con pochissimi esempi da cui imparare),
# assumiamo che tutti i giorni provengano dallo stesso processo generatore
# di mercato e li mettiamo in un unico pool per l'addestramento.
#
# Le sequenze vengono comunque costruite SOLO all'interno di ciascun
# giorno (mai a cavallo tra un giorno e il successivo), per non creare
# transizioni artificiali; è il pool finale di sequenze ad essere condiviso.

def load_all_days(data_dir, seq_length):
    """Carica tutti i CSV validi nella cartella e ne ritorna i dati grezzi."""
    search_path = os.path.join(data_dir, "*.csv")
    file_list = sorted(glob.glob(search_path))

    days = []
    for file_path in tqdm(file_list, desc="Caricamento CSV"):
        filename = os.path.basename(file_path)
        date_part = filename.replace(".csv", "").replace("trade_", "")
        try:
            trades_df = pd.read_csv(file_path, header=None)
        except Exception as e:
            print(f"Errore durante il caricamento di {filename}: {e}")
            continue

        if len(trades_df) <= seq_length:
            print(f"File {filename} troppo corto ({len(trades_df)} righe). Escluso dal pool.")
            continue

        data_array = trades_df[[0, 1, 2, 3]].to_numpy()
        days.append({
            'filename': filename,
            'date_part': date_part,
            'file_path': file_path,
            'data_array': data_array
        })

    return days


def fit_global_scaler(days):
    """Adatta un unico MinMaxScaler sui dati continui (tempo, prezzo, volume) di tutti i giorni."""
    all_cont = np.vstack([d['data_array'][:, :3] for d in days])
    scaler = MinMaxScaler()
    scaler.fit(all_cont)
    return scaler


def _count_sequences_for_day(n_rows, seq_length, stride):
    """Numero di sequenze che range(0, n_rows - seq_length, stride) produrrebbe."""
    n = n_rows - seq_length
    if n <= 0:
        return 0
    return (n + stride - 1) // stride  # ceil(n / stride)


def build_pooled_sequences(days, scaler, seq_length, stride=1):
    """
    Costruisce le sequenze di training usando lo scaler globale, mantenendo
    ogni sequenza interamente dentro un singolo giorno, e le mette tutte
    in un unico pool.

    'stride' > 1 permette di ridurre il numero di sequenze sovrapposte,
    utile se il pool totale diventa troppo grande per la memoria disponibile.

    NOTA SULLA MEMORIA: invece di accumulare le sequenze in una lista Python
    e poi fare np.array(lista) + torch.tensor(...) (due copie extra dell'intero
    pool, da cui il picco di RAM durante il caricamento), pre-allochiamo subito
    un unico array numpy della dimensione finale esatta e lo riempiamo in place.
    torch.from_numpy() poi non copia i dati, ma condivide lo stesso buffer.
    """
    n_features = 4

    # Primo passaggio (economico): calcoliamo quante sequenze totali servono,
    # per poter allocare un solo blocco di memoria della dimensione giusta.
    counts = [_count_sequences_for_day(len(d['data_array']), seq_length, stride) for d in days]
    total_sequences = sum(counts)

    print(f"Allocazione di un array per {total_sequences} sequenze "
          f"({seq_length} step x {n_features} feature)...")
    sequences_np = np.empty((total_sequences, seq_length, n_features), dtype=np.float32)

    idx = 0
    for d, n_seq_day in tqdm(zip(days, counts), total=len(days), desc="Costruzione sequenze"):
        if n_seq_day == 0:
            continue

        data_array = d['data_array']
        scaled_cont = scaler.transform(data_array[:, :3])
        scaled_signs = np.where(data_array[:, 3] == 1, 1.0, 0.0).reshape(-1, 1)
        dataset_scaled = np.hstack([scaled_cont, scaled_signs]).astype(np.float32)

        for j, i in enumerate(range(0, len(dataset_scaled) - seq_length, stride)):
            sequences_np[idx + j] = dataset_scaled[i:i + seq_length]
        idx += n_seq_day

    sequences = torch.from_numpy(sequences_np)
    return sequences


# =====================================================================
# 3. FUNZIONE DI ADDESTRAMENTO (POOL GLOBALE + INNESTO PRE-TRAINED)
# =====================================================================
#
# Modifiche rispetto alla versione originaria, pensate per due problemi:
#   (a) il discriminatore diventava troppo bravo troppo in fretta e il
#       generatore smetteva di ricevere un gradiente utile;
#   (b) i dati sintetici risultavano statisticamente troppo lontani dai
#       dati reali (anche a causa del training su un solo giorno alla volta).
#
# Soluzioni introdotte:
#   1. Addestramento su un pool di sequenze provenienti da TUTTI i giorni
#      (vedi funzioni sopra), invece che su un giorno isolato
#   2. Label smoothing (le etichette "reali" sono 0.9 invece di 1.0)
#   3. Instance noise: rumore gaussiano aggiunto agli input del
#      discriminatore, che decresce epoca dopo epoca
#   4. Aggiornamento del discriminatore "condizionato": se la sua accuratezza
#      media supera una soglia, il suo passo di ottimizzazione viene saltato
#      per quel batch (continua comunque ad allenarsi il generatore)
#   5. Feature matching loss: il generatore viene penalizzato anche in base
#      alla distanza tra media/deviazione standard dei dati reali e di
#      quelli sintetici, così da avvicinare le due distribuzioni non solo
#      "ingannando" il discriminatore ma anche nei valori aggregati

def train_gan(sequences, epochs=5, batch_size=128, latent_dim=10,
              pretrained_gen=None, pretrained_disc=None,
              d_acc_threshold=0.85, label_smoothing=0.9,
              instance_noise_start=0.15, lambda_fm=5.0):
    """
    'sequences' è un tensore [N, seq_length, 4] già scalato, tipicamente
    prodotto da build_pooled_sequences().
    """
    seq_length = sequences.shape[1]
    dataloader = DataLoader(TensorDataset(sequences), batch_size=batch_size, shuffle=True)

    # Inizializzazione modelli (Transfer Learning se già esistenti)
    if pretrained_gen is not None:
        generator = pretrained_gen
    else:
        generator = LOBGenerator(latent_dim, output_dim=4).to(device)

    if pretrained_disc is not None:
        discriminator = pretrained_disc
    else:
        discriminator = LOBDiscriminator(input_dim=4).to(device)

    # Loss e Ottimizzatori
    criterion = nn.BCELoss()
    # Riduciamo di un fattore 4 il learning rate del Discriminatore
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loop di addestramento
    generator.train()
    discriminator.train()
    epoch_bar = tqdm(range(epochs), desc="Epoche", unit="epoca")
    for epoch in epoch_bar:
        # L'instance noise decresce linearmente fino a 0 man mano che
        # l'addestramento procede: aiuta soprattutto nelle prime epoche,
        # quando il discriminatore rischia di "imparare a memoria".
        noise_std = instance_noise_start * max(0.0, 1.0 - epoch / max(1, epochs - 1))

        batch_bar = tqdm(dataloader, desc=f"  Epoca {epoch+1}/{epochs}", leave=False, unit="batch")
        for i, (real_seqs,) in enumerate(batch_bar):
            real_seqs = real_seqs.to(device)
            current_batch_size = real_seqs.size(0)

            real_labels = torch.full((current_batch_size, 1), label_smoothing, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            # Genera dati falsi dal rumore (serve sia per D che per G)
            z = torch.randn(current_batch_size, seq_length, latent_dim, device=device)
            fake_seqs = generator(z)

            # Aggiungiamo instance noise sia ai dati reali che a quelli falsi
            if noise_std > 0:
                real_input = real_seqs + torch.randn_like(real_seqs) * noise_std
                fake_input_for_d = fake_seqs.detach() + torch.randn_like(fake_seqs) * noise_std
            else:
                real_input = real_seqs
                fake_input_for_d = fake_seqs.detach()

            # --- Valutiamo l'accuratezza corrente del discriminatore ---
            with torch.no_grad():
                real_pred = discriminator(real_input)
                fake_pred = discriminator(fake_input_for_d)
                real_acc = (real_pred > 0.5).float().mean().item()
                fake_acc = (fake_pred < 0.5).float().mean().item()
                d_acc = (real_acc + fake_acc) / 2.0

            # --- Addestramento Discriminatore (condizionato) ---
            # Se il discriminatore è già troppo bravo, saltiamo il suo
            # aggiornamento per questo batch, lasciando respirare il generatore.
            if d_acc < d_acc_threshold:
                d_optimizer.zero_grad()

                outputs_real = discriminator(real_input)
                d_loss_real = criterion(outputs_real, real_labels)

                outputs_fake = discriminator(fake_input_for_d)
                d_loss_fake = criterion(outputs_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
            else:
                # Calcoliamo comunque la loss solo per il logging, senza backward
                with torch.no_grad():
                    d_loss = criterion(discriminator(real_input), real_labels) + \
                              criterion(discriminator(fake_input_for_d), fake_labels)

            # --- Addestramento Generatore ---
            g_optimizer.zero_grad()

            outputs = discriminator(fake_seqs)
            adv_loss = criterion(outputs, torch.ones(current_batch_size, 1, device=device))

            # Feature matching: avviciniamo media e deviazione standard
            # delle sequenze generate a quelle reali (per singola feature)
            real_mean = real_seqs.mean(dim=(0, 1))
            fake_mean = fake_seqs.mean(dim=(0, 1))
            real_std = real_seqs.std(dim=(0, 1))
            fake_std = fake_seqs.std(dim=(0, 1))
            fm_loss = torch.mean((real_mean - fake_mean) ** 2) + torch.mean((real_std - fake_std) ** 2)

            g_loss = adv_loss + lambda_fm * fm_loss
            g_loss.backward()
            g_optimizer.step()

            # Aggiorniamo la progress bar del batch con le metriche correnti,
            # cosi si vede l'andamento in tempo reale e non solo a fine epoca.
            batch_bar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
                "D_acc": f"{d_acc:.2f}",
            })

        epoch_bar.set_postfix({
            "D_loss": f"{d_loss.item():.4f}",
            "G_loss": f"{g_loss.item():.4f}",
            "D_acc": f"{d_acc:.2f}",
            "noise": f"{noise_std:.3f}",
        })
        tqdm.write(f"   Epoca [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | "
                   f"G Loss: {g_loss.item():.4f} | D Acc: {d_acc:.2f} | Noise: {noise_std:.3f}")

    return generator, discriminator


# =====================================================================
# 4. FUNZIONE DI GENERAZIONE E POST-PROCESSING (OPZIONE A)
# =====================================================================

def generate_synthetic_data(generator, scaler, num_trades_to_generate, seq_length=100, latent_dim=10, output_path='synthetic_trades.csv'):
    generator.eval()

    # Calcoliamo quanti blocchi sequenziali generare per raggiungere il numero totale richiesto
    num_blocks = int(np.ceil(num_trades_to_generate / seq_length))
    all_generated_blocks = []

    with torch.no_grad():
        for _ in range(num_blocks):
            z = torch.randn(1, seq_length, latent_dim).to(device)
            generated_seq = generator(z).cpu().numpy()[0]
            all_generated_blocks.append(generated_seq)

    # Uniamo tutti i blocchi generati in un unico dataset
    synthetic_dataset = np.vstack(all_generated_blocks)[:num_trades_to_generate]

    # POST-PROCESSING (Opzione A)
    generated_cont = synthetic_dataset[:, :3]
    generated_sign_probs = synthetic_dataset[:, 3]

    # Denormalizziamo tempo, prezzo e volume usando lo scaler originale
    denormalized_cont = scaler.inverse_transform(generated_cont)

    # Convertiamo la probabilità continua nel segno discreto +1 o -1 (Soglia a 0.5)
    discrete_signs = np.where(generated_sign_probs > 0.5, 1, -1)

    # Ricostruzione del DataFrame finale nello stesso ordine del tuo script originale
    df_synthetic = pd.DataFrame({
        'time_delta': denormalized_cont[:, 0],
        'price': denormalized_cont[:, 1],
        'volume': denormalized_cont[:, 2],
        'sign': discrete_signs
    })

    # Arrotondiamo e puliamo i dati per renderli conformi alla microstruttura reale
    df_synthetic['volume'] = np.abs(df_synthetic['volume'].round()).astype(int)
    df_synthetic['time_delta'] = np.sort(np.abs(df_synthetic['time_delta']))  # Forza l'ordine incrementale del tempo

    # Salvataggio in formato CSV senza intestazioni né indice
    df_synthetic.to_csv(output_path, header=False, index=False)
    print(f"-> File sintetico salvato con successo in: {output_path}")


def _load_checkpoint_if_present(gen_ckpt_path, disc_ckpt_path, latent_dim):
    if not os.path.exists(gen_ckpt_path):
        return None, None
    print("Checkpoint trovato: riprendo l'addestramento da dove eri rimasto...")
    gen = LOBGenerator(latent_dim, output_dim=4).to(device)
    gen.load_state_dict(torch.load(gen_ckpt_path, map_location=device))
    disc = LOBDiscriminator(input_dim=4).to(device)
    disc.load_state_dict(torch.load(disc_ckpt_path, map_location=device))
    return gen, disc


# =====================================================================
# 5a. MODALITA' DI REVISIONE (TRAINING_MODE = True)
# =====================================================================
#
# Allena la rete sul POOL di sequenze provenienti da tutti i giorni
# disponibili, riprendendo sempre dall'ultimo checkpoint salvato su disco,
# e genera un solo file di "prova" da ispezionare. Ogni volta che rilanci
# lo script in questa modalità, l'addestramento riparte da dove era
# arrivato (transfer learning) invece che da zero.

def run_review_round(data_dir='database\\data', checkpoint_dir='database\\checkpoints',
                      review_dir='database\\review', seq_length=100, latent_dim=10,
                      epochs_per_round=5, stride=1):
    for folder in [checkpoint_dir, review_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    days = load_all_days(data_dir, seq_length)
    if not days:
        print(f"Errore: nessun file CSV valido trovato in '{data_dir}'.")
        return

    print(f"Pool di addestramento: {len(days)} giorni.")
    scaler = fit_global_scaler(days)
    sequences = build_pooled_sequences(days, scaler, seq_length, stride=stride)
    print(f"Totale sequenze nel pool: {sequences.shape[0]}")

    gen_ckpt_path = os.path.join(checkpoint_dir, 'latest_generator.pt')
    disc_ckpt_path = os.path.join(checkpoint_dir, 'latest_discriminator.pt')
    round_counter_path = os.path.join(checkpoint_dir, 'review_rounds.txt')

    current_gen, current_disc = _load_checkpoint_if_present(gen_ckpt_path, disc_ckpt_path, latent_dim)
    if current_gen is None:
        print("Nessun checkpoint trovato: parto da zero.")

    generator, discriminator = train_gan(
        sequences=sequences,
        epochs=epochs_per_round,
        batch_size=128,
        latent_dim=latent_dim,
        pretrained_gen=current_gen,
        pretrained_disc=current_disc
    )

    torch.save(generator.state_dict(), gen_ckpt_path)
    torch.save(discriminator.state_dict(), disc_ckpt_path)

    # Teniamo un contatore informativo di quante volte è stato rilanciato il round
    round_num = 1
    if os.path.exists(round_counter_path):
        with open(round_counter_path, 'r') as f:
            round_num = int(f.read().strip() or 0) + 1
    with open(round_counter_path, 'w') as f:
        f.write(str(round_num))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    review_path = os.path.join(review_dir, f'review_{timestamp}.csv')
    generate_synthetic_data(
        generator=generator,
        scaler=scaler,
        num_trades_to_generate=min(len(days[0]['data_array']), 50000),
        seq_length=seq_length,
        latent_dim=latent_dim,
        output_path=review_path
    )

    print("\n" + "=" * 60)
    print(f"Round di revisione #{round_num} completato ({epochs_per_round} epoche sul pool di {len(days)} giorni).")
    print(f"Ispeziona il file: {review_path}")
    print("Se non sei soddisfatto, rilancia semplicemente lo script: l'addestramento")
    print("riprenderà da questo checkpoint invece che ricominciare da zero.")
    print("Quando sei soddisfatto, imposta TRAINING_MODE = False per generare tutti i dati.")
    print("=" * 60)


# =====================================================================
# 5b. ADDESTRAMENTO SUL POOL + GENERAZIONE PER OGNI GIORNO (TRAINING_MODE = False)
# =====================================================================

def train_and_generate_all_days(data_dir='database\\data', output_dir='data_timegan',
                                 checkpoint_dir='checkpoints', seq_length=100, latent_dim=10,
                                 epochs=20, stride=1):
    """
    Allena UN SOLO modello condiviso sul pool di sequenze di tutti i giorni,
    poi genera un file sintetico per ciascun giorno (stesso numero di righe
    del giorno reale corrispondente), riusando lo stesso generatore e lo
    stesso scaler globale per tutti.
    """
    for folder in [output_dir, checkpoint_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    days = load_all_days(data_dir, seq_length)
    if not days:
        print(f"Errore: nessun file CSV valido trovato in '{data_dir}'.")
        return

    print(f"Trovati {len(days)} giorni validi. Costruzione del pool di sequenze...")
    scaler = fit_global_scaler(days)
    sequences = build_pooled_sequences(days, scaler, seq_length, stride=stride)
    print(f"Pool totale: {sequences.shape[0]} sequenze da {seq_length} step ciascuna.")

    gen_ckpt_path = os.path.join(checkpoint_dir, 'latest_generator.pt')
    disc_ckpt_path = os.path.join(checkpoint_dir, 'latest_discriminator.pt')

    current_gen, current_disc = _load_checkpoint_if_present(gen_ckpt_path, disc_ckpt_path, latent_dim)
    if current_gen is None:
        print("Nessun checkpoint trovato. Avvio addestramento da zero sul pool completo...")
    else:
        print("Checkpoint precedente caricato. Continuo l'addestramento sul pool completo...")

    generator, discriminator = train_gan(
        sequences=sequences,
        epochs=epochs,
        batch_size=128,
        latent_dim=latent_dim,
        pretrained_gen=current_gen,
        pretrained_disc=current_disc
    )

    torch.save(generator.state_dict(), gen_ckpt_path)
    torch.save(discriminator.state_dict(), disc_ckpt_path)
    print("Pesi aggiornati e salvati nella cartella checkpoints.")

    print("\n" + "=" * 60)
    print("Addestramento sul pool completato. Genero i file sintetici, giorno per giorno...")

    for d in days:
        output_filename = f"synthetic_{d['date_part']}.csv"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Giorno {d['date_part']} già elaborato. Salto al file successivo.")
            continue

        generate_synthetic_data(
            generator=generator,
            scaler=scaler,
            num_trades_to_generate=len(d['data_array']),
            seq_length=seq_length,
            latent_dim=latent_dim,
            output_path=output_path
        )

    print("\n" + "=" * 60)
    print("PROCESSO COMPLETATO! Trovi tutti i file generati in 'data_timegan'.")


# =====================================================================
# INTERFACCIA DI AVVIO
# =====================================================================
if __name__ == "__main__":
    # --- INTERRUTTORE PRINCIPALE ---
    # True  -> modalità di revisione: allena sul pool di tutti i giorni,
    #          genera UN SOLO file di prova e si ferma. Rilancia lo script
    #          tutte le volte che vuoi: riprende dal checkpoint precedente.
    # False -> modalità di produzione: usa l'ultimo checkpoint disponibile,
    #          allena ulteriormente sul pool completo e genera i dati
    #          sintetici per ogni giorno.
    TRAINING_MODE = True

    # 'stride' controlla quante sequenze sovrapposte vengono create per ogni
    # giorno durante il pooling. stride=1 usa tutte le sequenze possibili
    # (massima qualità, massimo uso di memoria); aumentalo (es. 5 o 10) se
    # il pool totale non entra in memoria con molti giorni di dati.
    STRIDE = 10

    if TRAINING_MODE:
        run_review_round(
            data_dir='database\\data',
            checkpoint_dir='database\\checkpoints',
            review_dir='database\\review',
            seq_length=100,
            latent_dim=10,
            epochs_per_round=1,  # quante epoche in più ad ogni rilancio dello script
            stride=STRIDE
        )
    else:
        train_and_generate_all_days(
            data_dir='database\\data',
            output_dir='database\\data_timegan',
            checkpoint_dir='database\\checkpoints',
            seq_length=100,
            latent_dim=10,
            epochs=20,
            stride=STRIDE
        )