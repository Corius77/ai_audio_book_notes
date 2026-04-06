#include <torch/script.h> // One-stop header.
#include "AudioFile/AudioFile.h"

#include <iostream>
#include <memory>


/**
 * Funkcja przekształcająca macierz (AudioBuffer) na płaski wektor 
 */
template <typename T>
std::vector<T> interleaveAudio (const std::vector<std::vector<T>>& buffer)
{
    // Sprawdzenie, czy bufor nie jest pusty, aby uniknąć błędów dostępu
    if (buffer.empty() || buffer[0].empty()) 
        return {};

    // Liczbę kanałów i próbek pobieramy bezpośrednio z rozmiarów wektorów
    size_t numChannels = buffer.size();
    size_t numSamples = buffer[0].size();

    std::vector<T> interleavedSamples;
    interleavedSamples.resize (numChannels * numSamples);

    for (size_t i = 0; i < numSamples; i++)
    {
        for (size_t channel = 0; channel < numChannels; channel++)
        {
            interleavedSamples[(i * numChannels) + channel] = buffer[channel][i];
        }
    }

    return interleavedSamples;
}

/**
 * Funkcja przekształcająca płaski wektor z przeplotem (interleaved) 
 * z powrotem na strukturę kanałów (AudioBuffer).
 */
template <typename T>
typename AudioFile<T>::AudioBuffer deinterleaveAudio (const std::vector<T>& interleavedData, int numChannels)
{
    if (numChannels <= 0 || interleavedData.empty())
        return {};

    // Obliczamy liczbę próbek przypadającą na jeden kanał
    int numSamplesPerChannel = static_cast<int> (interleavedData.size() / numChannels);

    // Tworzymy strukturę AudioBuffer (wektor wektorów)
    typename AudioFile<T>::AudioBuffer buffer;
    buffer.resize (numChannels);

    for (int channel = 0; channel < numChannels; channel++)
    {
        buffer[channel].resize (numSamplesPerChannel);
    }

    // Rozdzielamy próbki z płaskiego wektora do odpowiednich kanałów
    for (int i = 0; i < numSamplesPerChannel; i++)
    {
        for (int channel = 0; channel < numChannels; channel++)
        {
            // Odwrócenie wzoru indeksowania: (indeks_czasowy * liczba_kanałów) + kanał
            buffer[channel][i] = interleavedData[(i * numChannels) + channel];
        }
    }

    return buffer;
}

int main() { 
  torch::jit::script::Module my_lstm;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    my_lstm = torch::jit::load("my_lstm.pt");

    // Loading WAV file
    AudioFile<float> audioFile;
    audioFile.load("sine_400_16bit.wav");

    std::vector<float> signal = interleaveAudio(audioFile.samples);

    // Converting float vector to tensor
    // Without re-allocating memory
    torch::Tensor in_t = torch::from_blob(signal.data(), {static_cast<int64_t> (signal.size())});

    // Reshaping from [1, 2, 3] to [[1], [2], [3]]
    in_t = in_t.view({-1, 1});

    // Tensor to value vector
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(in_t);

    // inference
    torch::jit::IValue out_ival = my_lstm.forward(inputs);

    auto out_elements = out_ival.toTuple()->elements();

    torch::Tensor out_t = out_elements[0].toTensor();

    out_t = out_t.view({-1});

    float* data_ptr = out_t.data_ptr<float>();
    std::vector<float> data_vector(data_ptr, data_ptr + out_t.numel());

    // Saving the inference on input to WAV
    int numChannels = audioFile.getNumChannels();
    std::vector<std::vector<float>> data_vector1 = deinterleaveAudio(data_vector, numChannels);
    bool ok = audioFile.setAudioBuffer(data_vector1);
    audioFile.save("test.wav");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}