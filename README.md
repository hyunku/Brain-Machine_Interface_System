## 지도교수님

김훈희 교수님

## 팀원

강현구, 김묘경, 이회영, 추연희

## 프로젝트 기간

2022.3 ~ 2022.9 <br><br>

# Mind Pong

system : EEG -> decoder (server) -> Pong

EEG
 - OpenBCI, CYTON_DAISY board connected with serial port

decoder
 - preprocessing with mne package, inference with Echo State Network (ESN) or Linear Model

Pong

 - EEG training/testing sessions
 - and Pong (based on https://github.com/clear-code-projects/Pong_in_Pygame/blob/master/pong11_sprites.py)

## install

git clone https://github.com/4NBrain/pong.git

cd pong

pip install -r requirements.txt

## settings

app is controlled by config.json

make sure proper serial_port (eg.: "COM4") before running src/run_eeg.py

choose pretrain model (linear, esn) on config.json

## run

running decoder

 - python src/run_decoder_2.py
  
running pong app

 - python src/run_app.py
  
running eeg app

 - python src/run_eeg.py
