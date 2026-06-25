import os
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

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
# 2. FUNZIONE DI ADDESTRAMENTO (CON INNESTO PRE-TRAINED)
# =====================================================================

def train_gan(data_array, seq_length=100, epochs=5, batch_size=64, latent_dim=10, pretrained_gen=None, pretrained_disc=None):
    # 1. Pre-processing: Normalizziamo i dati tra 0 e 1 (tranne il segno mappato 0-1)
    scaler = MinMaxScaler()
    scaled_cont = scaler.fit_transform(data_array[:, :3])
    
    # Trasformiamo i segni da [-1, 1] a [0, 1] per coerenza matematica con la Sigmoide
    scaled_signs = np.where(data_array[:, 3] == 1, 1.0, 0.0).reshape(-1, 1)
    dataset_scaled = np.hstack([scaled_cont, scaled_signs])
    
    # Creazione delle sequenze temporali per l'addestramento
    sequences = []
    for i in range(len(dataset_scaled) - seq_length):
        sequences.append(dataset_scaled[i:i+seq_length])
    sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
    
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
    for epoch in range(epochs):
        for i, (real_seqs,) in enumerate(dataloader):
            real_seqs = real_seqs.to(device)
            current_batch_size = real_seqs.size(0)
            
            # --- Addestramento Discriminatore ---
            d_optimizer.zero_grad()
            
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)
            
            # Loss sui dati reali
            outputs = discriminator(real_seqs)
            d_loss_real = criterion(outputs, real_labels)
            
            # Genera dati falsi dal rumore
            z = torch.randn(current_batch_size, seq_length, latent_dim).to(device)
            fake_seqs = generator(z)
            
            # Loss sui dati falsi
            outputs = discriminator(fake_seqs.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # --- Addestramento Generatore ---
            g_optimizer.zero_grad()
            outputs = discriminator(fake_seqs)
            g_loss = criterion(outputs, real_labels) 
            
            g_loss.backward()
            g_optimizer.step()
            
        print(f"   Epoca [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        
    return generator, discriminator, scaler


# =====================================================================
# 3. FUNZIONE DI GENERAZIONE E POST-PROCESSING (OPZIONE A)
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
    df_synthetic['time_delta'] = np.sort(np.abs(df_synthetic['time_delta'])) # Forza l'ordine incrementale del tempo
    
    # Salvataggio in formato CSV senza intestazioni né indice
    df_synthetic.to_csv(output_path, header=False, index=False)
    print(f"-> File sintetico salvato con successo in: {output_path}")


# =====================================================================
# 4. PIPELINE DI COORDINAMENTO ANNUALE (INCREMENTALE)
# =====================================================================

def process_and_generate_year_data(data_dir='database\\data', output_dir='data_timegan', checkpoint_dir='checkpoints', seq_length=100, latent_dim=10):
    """
    Scansiona la cartella dati reali, gestisce l'addestramento incrementale e permette
    di interrompere/riprendere l'esecuzione in qualsiasi momento senza perdere dati.
    """
    # Creazione delle cartelle necessarie
    for folder in [output_dir, checkpoint_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Trova tutti i file CSV nella cartella dei dati reali
    search_path = os.path.join(data_dir, "*.csv")
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"Errore: Nessun file CSV trovato in '{data_dir}'. Controlla il percorso.")
        return

    print(f"Trovati {len(file_list)} file da elaborare per l'intero anno.")

    # Variabili per tenere traccia dello stato delle reti in memoria RAM
    current_gen = None
    current_disc = None
    
    # Definiamo i percorsi fissi su disco per i checkpoint di ripristino d'emergenza
    gen_ckpt_path = os.path.join(checkpoint_dir, 'latest_generator.pt')
    disc_ckpt_path = os.path.join(checkpoint_dir, 'latest_discriminator.pt')

    # Iterazione giorno per giorno
    for file_path in file_list:
        filename = os.path.basename(file_path)
        
        # Estrazione della data dal nome del file (es. trade_2026_06_01.csv -> 2026_06_01)
        date_part = filename.replace(".csv", "").replace("trade_", "")
        
        # Costruiamo il percorso del file finale atteso
        output_filename = f"synthetic_{date_part}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # --- CONTROLLO DI AVANZAMENTO: Salta se già generato ---
        if os.path.exists(output_path):
            print(f"Giorno {date_part} già elaborato. Salto al file successivo.")
            continue
            
        print(f"\n" + "="*60)
        print(f"INIZIO ELABORAZIONE FILE REALE: {filename}")
        
        # Caricamento dei dati del singolo giorno
        try:
            trades_df = pd.read_csv(file_path, header=None)
            if len(trades_df) <= seq_length:
                print(f"File {filename} troppo corto ({len(trades_df)} righe). Salto.")
                continue
                
            data_array = trades_df[[0, 1, 2, 3]].to_numpy()
            num_trades_to_generate = len(data_array)
            
        except Exception as e:
            print(f"Errore durante il caricamento di {filename}: {e}")
            continue

        # --- GESTIONE DEI CHECKPOINT DA DISCO: In caso di ripartenza dopo interruzione ---
        if current_gen is None and os.path.exists(gen_ckpt_path):
            print("Ripristino dei modelli dall'ultimo checkpoint su disco rilevato...")
            current_gen = LOBGenerator(latent_dim, output_dim=4).to(device)
            current_gen.load_state_dict(torch.load(gen_ckpt_path, map_location=device))
            
            current_disc = LOBDiscriminator(input_dim=4).to(device)
            current_disc.load_state_dict(torch.load(disc_ckpt_path, map_location=device))
            print("Modelli ripristinati con successo.")

        # --- TRANSFER LEARNING: Regolazione delle epoche ---
        # Se la rete parte da zero (primo file in assoluto), fa 20 epoche per capire le regole del LOB.
        # Nei giorni successivi bastano 4 epoche perché deve solo adattarsi ai piccoli cambiamenti di prezzo/volume.
        if current_gen is None:
            epochs_to_run = 20
            print("Nessun modello precedente trovato. Avvio addestramento da zero (Richiede più tempo)...")
        else:
            epochs_to_run = 4
            print("Modello precedente caricato. Adattamento incrementale in corso...")

        # Esecuzione dell'addestramento
        generator, discriminator, scaler = train_gan(
            data_array=data_array,
            seq_length=seq_length,
            epochs=epochs_to_run,
            batch_size=128,  # Batch_size ottimizzato per velocità su GPU
            latent_dim=latent_dim,
            pretrained_gen=current_gen,
            pretrained_disc=current_disc
        )
        
        # Aggiorniamo i riferimenti in memoria RAM per il ciclo successivo
        current_gen = generator
        current_disc = discriminator
        
        # SALVATAGGIO DI SICUREZZA DEI PESI SU HARD DISK
        torch.save(generator.state_dict(), gen_ckpt_path)
        torch.save(discriminator.state_dict(), disc_ckpt_path)
        print("Pesi aggiornati e salvati nella cartella checkpoints.")

        # Generazione effettiva dei dati sintetici e scrittura del CSV
        generate_synthetic_data(
            generator=generator,
            scaler=scaler,
            num_trades_to_generate=num_trades_to_generate,
            seq_length=seq_length,
            latent_dim=latent_dim,
            output_path=output_path
        )
        
        # Pulizia della cache della memoria per prevenire crash su lunghe sessioni
        del data_array, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("PROCESSO ANNUALE COMPLETATO! Trovi tutti i file generati in 'data_timegan'.")


# =====================================================================
# INTERFACCIA DI AVVIO
# =====================================================================
if __name__ == "__main__":
    # Avvia la pipeline principale
    process_and_generate_year_data(
        data_dir='database\\data',
        output_dir='database\\data_timegan',
        checkpoint_dir='database\\checkpoints',
        seq_length=100,
        latent_dim=10
    )