Fine-tuning LLaMA-2 on Hawaii Wildfire Dataset

This project shows how to fine-tune the LLaMA-2 7B Chat model using a small text dataset about the Hawaii wildfires.
The goal is to teach the model how to answer questions related to the topic with better accuracy and context.

The fine-tuning is done using LoRA (Low-Rank Adaptation) and 4-bit quantization so it can run on smaller GPUs.

📌 Features

1]Loads the LLaMA-2 7B Chat model in 4-bit using bitsandbytes.
2]Applies LoRA adapters to reduce training cost.
3]Uses a small text dataset (hawaii_wf_2.txt) for training.
4]Trains with the Hugging Face Trainer class.
5]Saves LoRA checkpoints for later use.
6]Runs inference using the fine-tuned model.

📂 Project Structure
Finetuning_llama2_Hawaii_data/
├── README.md
├── Finetuning_personal_data.ipynb   
├── data/
│   └── hawaii_wf_2.txt              
└── finetunedModel/     

🚀 How the Fine-Tuning Works 
  Load LLaMA-2 7B Chat
  Downloaded from Hugging Face (requires access).
  Load tokenizer
  Makes text into tokens the model can understand.
  Prepare the model for 4-bit training
  Saves GPU memory using BitsAndBytesConfig.
  Apply LoRA
  Add small trainable layers on top of the frozen model.
  Load training data
  Hawaii wildfire dataset (plain text).
  Tokenize the data
  Convert all text into token IDs.
  Train the LoRA adapters
  Only a small number of parameters are updated.
  Save and reload the model
  Base model + LoRA adapters = fine-tuned output.
  Ask questions
  The model gives better wildfire-related answers after training. 

📦 Requirements
Install the required libraries: 
  pip install transformers
  pip install datasets
  pip install peft
  pip install accelerate
  pip install bitsandbytes
  pip install GPUtil

▶️ Running the Notebook
  Open Finetuning_personal_data.ipynb in Google Colab or Jupyter.
  Run each cell step-by-step.
  After training, checkpoints will be saved in finetunedModel/.
  Use the last notebook cell to test the model with a question.
  
