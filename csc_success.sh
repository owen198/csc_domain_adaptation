
# syn->target model
python csc_transformer.py W4662FM0605 W4662FM0606 30 16 128 64
python csc_transformer.py W4662FM0605 W4633070102 30 16 128 64
python csc_transformer.py W4662FM0606 W4662FM0605 30 4 128 64
python csc_transformer.py W4633070102 W4662FM0605 30 4 128 64

# target->syn model
python csc_transformer.py W4633070102 W4662FM0605 30 4 128 64
