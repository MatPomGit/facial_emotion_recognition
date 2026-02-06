[![GitHub issues](https://img.shields.io/github/issues/jhan15/facial_emotion_recognition)](https://github.com/jhan15/facial_emotion_recognition/issues)
![GitHub last commit](https://img.shields.io/github/last-commit/jhan15/facial_emotion_recognition?color=ff69b4)

# Rozpoznawanie Emocji na Twarzy (facial_emotion_recognition)

## Wprowadzenie
Witaj w projekcie rozpoznawania emocji na twarzy! Ten projekt ma na celu nauczenie Cię, jak wykorzystać uczenie maszynowe do automatycznego rozpoznawania ludzkich emocji na podstawie zdjęć twarzy.

### Co robi ten projekt?
System rozpoznaje emocje na twarzy w 7 kategoriach:
- **Angry** (Złość) - gdy ktoś jest zdenerwowany lub wściekły
- **Disgust** (Obrzydzenie) - reakcja na coś nieprzyjemnego
- **Fear** (Strach) - gdy ktoś się boi
- **Happy** (Radość) - uśmiech i pozytywne emocje
- **Sad** (Smutek) - gdy ktoś jest smutny lub przygnębiony
- **Surprise** (Zaskoczenie) - reakcja na coś nieoczekiwanego
- **Neutral** (Neutralność) - brak wyraźnych emocji

### Jak to działa?
Projekt składa się z dwóch głównych kroków:
1. **Detekcja twarzy** - najpierw znajdujemy twarze na zdjęciu
2. **Rozpoznawanie emocji** - następnie analizujemy wykryte twarze i określamy emocje

## Zbiór Danych
Projekt wykorzystuje zbiór danych pochodzący z konkursu, który jest bardzo podobny do znanego zbioru FER2013. 

### Co zawiera zbiór danych?
- Tysiące zdjęć twarzy w skali szarości
- Każde zdjęcie ma rozmiar 48x48 pikseli
- Każda twarz jest oznaczona jedną z 7 emocji

![image](https://user-images.githubusercontent.com/62132206/122208856-22e01000-cea4-11eb-8047-24e3a01e28f2.png)

**Uwaga dla studentów:** Zbiór danych nie jest zbalansowany - niektóre emocje (jak radość) występują częściej niż inne (jak obrzydzenie). To typowy problem w uczeniu maszynowym, który rozwiązujemy techniką upsamplingowania.

## Architektura Modelu

### Pipeline (Proces Przetwarzania)
Nasz system działa w dwóch etapach:

```bash
Pipeline
├── Detekcja twarzy: mediapipe (API Google)
└── Rozpoznawanie emocji: VGGNet (sieć neuronowa)
```

### 1. Detekcja Twarzy - mediapipe
Używamy API [mediapipe](https://github.com/google/mediapipe) od Google do znajdowania twarzy na zdjęciach.
- **Co to jest?** - Gotowe narzędzie Google do wykrywania twarzy
- **Dlaczego to używamy?** - Jest szybkie, dokładne i łatwe w użyciu
- **Co robi?** - Znajduje położenie twarzy na zdjęciu i wycina je

### 2. Rozpoznawanie Emocji - VGGNet
Do rozpoznawania emocji używamy sieci neuronowej w stylu VGG.

#### Architektura VGGNet
Nasza sieć składa się z 15 warstw:
- 8 warstw konwolucyjnych (Conv2D) - wydobywają cechy z obrazu
- 4 warstwy pooling (MaxPool2D) - zmniejszają rozmiar danych
- 3 warstwy fully connected (Dense) - podejmują finalną decyzję

![image](https://user-images.githubusercontent.com/62132206/122204954-004bf800-cea0-11eb-981b-c7b1cbb935fc.png)

**Dla początkujących:**
- **Warstwy konwolucyjne** - uczą się rozpoznawać proste cechy jak krawędzie, później bardziej złożone jak oczy czy usta
- **Warstwy pooling** - redukują rozmiar danych, zachowując najważniejsze informacje
- **Warstwy dense** - łączą wszystkie informacje i decydują o emocji

### Przepływ Danych (Data Flow)

![image](https://user-images.githubusercontent.com/62132206/122206734-dd224800-cea1-11eb-9670-19b718667bbd.png)

**Krok po kroku:**
1. Obraz wejściowy (48x48 pikseli)
2. Normalizacja (skalowanie wartości 0-255 do 0-1)
3. Warstwy konwolucyjne wydobywają cechy
4. Warstwy pooling zmniejszają rozmiar
5. Flatten - wyprostowanie danych do jednego wektora
6. Warstwy dense - klasyfikacja
7. Softmax - prawdopodobieństwa dla każdej emocji
8. Wynik - przewidziana emocja

### Upsampling (Nadpróbkowanie)
Używamy techniki SMOTE (Synthetic Minority Over-sampling Technique) do zbalansowania zbioru danych.

![image](https://user-images.githubusercontent.com/62132206/122207118-10fd6d80-cea2-11eb-98b1-b13e678be8f7.png)

**Co to znaczy?**
- Niektóre emocje występują rzadziej w zbiorze danych
- SMOTE tworzy syntetyczne (sztuczne) przykłady rzadkich emocji
- Dzięki temu model lepiej uczy się rozpoznawać wszystkie emocje

## Jak Używać Projektu

### Wymagania Wstępne
Przed rozpoczęciem upewnij się, że masz:
- Konto Google (do użycia Google Colab)
- Podstawową znajomość Pythona
- Chęć do nauki!

### Krok 1: Przygotowanie Danych
Jeśli masz własny zbiór danych, użyj notebooka `read_data.ipynb`:
- Wczytuje zdjęcia twarzy
- Przygotowuje etykiety (nazwy emocji)
- Zapisuje dane w odpowiednim formacie

### Krok 2: Trening Modelu
Użyj notebooka `training.ipynb` do wytrenowania własnego modelu [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhan15/facial_emotion_recognition/blob/master/training.ipynb)

**Co się dzieje podczas treningu?**
1. Wczytanie danych
2. Podział na zbiory: treningowy, walidacyjny, testowy
3. Upsampling danych treningowych
4. Budowa modelu VGGNet
5. Trening przez wiele epok (iteracji)
6. Zapisanie najlepszego modelu

**Wskazówki dla studentów:**
- Trening może zająć kilka godzin (użyj GPU w Colab!)
- Obserwuj wykresy - loss powinien maleć, accuracy rosnąć
- Jeśli model się nie uczy, spróbuj zmniejszyć learning rate

### Krok 3: Inference (Wnioskowanie)
Użyj notebooka `inference.ipynb` do testowania modelu [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jhan15/facial_emotion_recognition/blob/master/inference.ipynb)

**Co możesz zrobić?**
- Przeanalizować pojedyncze zdjęcie
- Przetworzyć wiele zdjęć na raz
- Rozpoznawać emocje w czasie rzeczywistym z kamery
- Przetwarzać wideo

### Wytrenowane Wagi Modelu
Gotowe wagi (parametry) wytrenowanego modelu znajdują się w katalogu [saved_models](https://github.com/jhan15/facial_emotion_recognition/tree/master/saved_models).

**Dwa dostępne modele:**
- `vggnet.h5` - wytrenowany na oryginalnych danych
- `vggnet_up.h5` - wytrenowany na danych po upsamplingowaniu

**Domyślna konfiguracja:** System używa voting classifier - łączy predykcje z obu modeli, co zwiększa dokładność!

**Dla ciekawskich:** Voting classifier działa jak głosowanie - jeśli oba modele są zgodne, to pewnie mają rację. Jeśli się różnią, wybieramy odpowiedź z wyższym prawdopodobieństwem.

## Wydajność

Poniżej znajduje się macierz pomyłek pokazująca, jak dobrze model rozpoznaje poszczególne emocje:

![image](https://user-images.githubusercontent.com/62132206/122231604-d94eef80-ceba-11eb-9ad4-1f73517da0ec.png)

**Jak czytać macierz pomyłek?**
- Wiersze = rzeczywiste emocje
- Kolumny = przewidziane emocje
- Przekątna = poprawne przewidzenia
- Poza przekątną = błędy (pomyłki)

**Obserwacje:**
- Model najlepiej radzi sobie z radością (happy)
- Czasami myli strach (fear) ze zaskoczeniem (surprise) - to normalne, bo są podobne!

## Przykładowe Wyniki

### Wideo w Czasie Rzeczywistym

<img src="https://user-images.githubusercontent.com/62132206/122562096-f7475c00-d042-11eb-9db0-f22fcba2720e.gif" width="500">

**Co widzisz?**
- Ramka wokół wykrytej twarzy
- Nazwa emocji nad ramką
- Różne kolory dla różnych emocji

### Przykładowe Zdjęcia

<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_1.jpg?raw=true" width="500">
<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_3.jpg?raw=true" width="500">
<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_5.jpg?raw=true" width="500">
<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_7.jpg?raw=true" width="500">
<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_8.jpg?raw=true" width="500">
<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_9.jpg?raw=true" width="500">
<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_12.jpg?raw=true" width="500">
<img src="https://github.com/jhan15/facial_emotion_recognition/blob/master/run/inference/out_17.jpg?raw=true" width="500">

## Dalsze Kroki i Eksperymenty

**Dla ambitnych studentów:**
1. Spróbuj dostroić hiperparametry (learning rate, batch size)
2. Dodaj więcej warstw do sieci
3. Użyj data augmentation (rotacje, przesunięcia)
4. Wypróbuj inne architektury (ResNet, EfficientNet)
5. Zbadaj, które emocje są najtrudniejsze do rozpoznania i dlaczego

## Problemy i Rozwiązania

**Najczęstsze problemy:**
- **Model się nie uczy** → Sprawdź learning rate, może być za duży
- **Brak GPU** → Użyj Google Colab z darmowym GPU
- **Overfitting** → Zwiększ dropout, użyj więcej danych
- **Słaba dokładność** → Spróbuj upsamplingu lub zwiększ rozmiar modelu

## Podziękowania

Projekt bazuje na oryginalnym repozytorium [jhan15/facial_emotion_recognition](https://github.com/jhan15/facial_emotion_recognition).

**Zasoby do nauki:**
- [Kurs Deep Learning](https://www.deeplearning.ai/)
- [Dokumentacja TensorFlow](https://www.tensorflow.org/)
- [Dokumentacja mediapipe](https://google.github.io/mediapipe/)
