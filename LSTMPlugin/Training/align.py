import numpy as np
import soundfile as sf
from scipy import signal

def align_24bit_wavs(input_path, processed_path, output_path):
    """
    Wyrównuje pliki audio i wymusza zapis w formacie PCM 24-bit.
    """
    # 1. Odczyt plików - sf.read domyślnie zwraca float32 w zakresie [-1, 1]
    # To najlepszy sposób na zachowanie precyzji 24 bitów podczas obliczeń
    data_in, samplerate_in = sf.read(input_path)
    data_out, samplerate_out = sf.read(processed_path)

    if samplerate_in != samplerate_out:
        raise ValueError("Częstotliwość próbkowania musi być identyczna!")

    # 2. Konwersja do mono na potrzeby korelacji
    def to_mono(data):
        return data[:, 0] if len(data.shape) > 1 else data

    sig_in = to_mono(data_in)
    sig_out = to_mono(data_out)

    # 3. Wykrywanie opóźnienia (Cross-correlation)
    correlation = signal.correlate(sig_out, sig_in, mode="full")
    lags = signal.correlation_lags(sig_out.size, sig_in.size, mode="full")
    delay = lags[np.argmax(correlation)]
    
    print(f"Wykryte opóźnienie: {delay} próbek")

    # 4. Wyrównanie sygnału wyjściowego
    if delay > 0:
        # Przetworzony sygnał jest opóźniony względem DI
        aligned_out = data_out[delay:]
    else:
        # Jeśli delay < 0, to DI jest opóźniony (rzadkie)
        aligned_out = data_out

    # 5. Zapis do pliku 24-bitowego
    # Kluczowy parametr: subtype='PCM_24'
    sf.write(output_path, aligned_out, samplerate_out, subtype='PCM_24')
    print(f"Zapisano wyrównany plik 24-bit: {output_path}")

if __name__ == "__main__":
    input = 'clean_signal'
    output = 'slipknot-signal'
    ext_ = '.wav'
    align_24bit_wavs(input+ext_, output+ext_, output+'_aligned'+ext_)