# end-to-end test with llama7b.
# TODO: improve to clean up last iter and make it take some params
python3 prepare_model.py
python3 test_gen.py
python3 finetune.py
python3 test_gen.py ./out/state_dict_19.pth
