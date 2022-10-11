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

make sure proper serial_port (eg.: "COM6") before running src/run_eeg.py

choose pretrain model (linear, esn) on config.json

## run

running decoder

 - python src/run_decoder_2.py
  
running pong app

 - python src/run_app.py
  
running eeg app

 - python src/run_eeg.py

## ingame option control

 difficulty option is controlled by socket_thread.py
 
 - check parameter named "difficulty"
 - "difficulty" parameter can get 0 ~ 1 value.
 - 0: 100% AI data, 0% eeg data -> easy mode
 - 0.5 : 50% AI data, 50% eeg data -> medium mode
 - 1: 0% AI data, 100% eeg data -> hard mode
 
 
 ball speed and bar speed option is controlled by run_app.py
 
 - check class named Player()
 - you can update ball speed by rewriting self.speed
 - you can update bar speed by rewriting self.speed
 - both bar and ball speed has default speed : 5
 
 ## Manual ( Update Scheduled )
 
 1. bci and computer is connected by socket make sure proper serial_port (eg.: "COM6") before running src/run_eeg.py
 2. run src/run_eeeg.py to collect your personal eeg data
 3. run src/run_app.py to collect your target data about your personal eeg data
 target data is signanl direction about your eeg data.
 signal value 1 means thinking about going up.
 signal value -1 means thinking about going down.
 signal value 0 means thinking about stay(nothing to do).
 
 4. 
