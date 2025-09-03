# FineTuning_QwenLLM_FromScratch

## 📜 Overview
This project is a comprehensive demonstration of fine-tuning and deploying a large language model (LLM) for specialized natural language processing (NLP) tasks, focusing on Arabic news analysis. It leverages the Qwen2.5-1.5B-Instruct model, a lightweight yet powerful LLM, fine-tuned to excel in two key tasks: structured data extraction and translation of Arabic news articles. The project combines cutting-edge tools like LLaMA-Factory for fine-tuning, vLLM for high-performance inference, and Gradio for an interactive user interface, making it a robust solution for processing and analyzing Arabic text.
🎯 Purpose
The primary goal is to adapt the Qwen model to handle Arabic news articles with high accuracy, extracting structured information (e.g., titles, keywords, summaries, categories, and named entities) or translating content into English. This is particularly useful for applications like automated journalism, content summarization, or cross-lingual information retrieval.
🔍 Key Components

Fine-Tuning Pipeline: Using LLaMA-Factory, the Qwen model is fine-tuned with custom datasets to specialize in Arabic NLP tasks. Fine-tuning enhances the model’s ability to understand and generate contextually relevant outputs for Arabic text, leveraging LoRA (Low-Rank Adaptation) for efficient parameter updates.
Tasks:

Extraction Details: Extracts structured JSON data from Arabic articles, including titles, keywords, summaries, categories (e.g., politics, art), and entities (e.g., people, locations, organizations), adhering to a strict Pydantic schema for validation.
Translation: Translates Arabic news articles into English, preserving meaning and context, with outputs formatted as JSON (title and content).


Deployment:

Gradio Interface: Provides an intuitive web UI for users to input Arabic text, select a task (extraction or translation), and view results in real-time.
vLLM Server: Enables efficient, scalable inference with support for LoRA adapters, ideal for high-throughput applications.
Load Testing: Uses Locust to simulate concurrent users, ensuring the model’s performance under stress.


Technical Approach: The project integrates modern NLP tools like Transformers, Pydantic for schema validation, and json_repair for robust JSON parsing. It also employs Google Colab for GPU-accelerated training and WandB for experiment tracking.

🌍 Applications
This system is designed for researchers, developers, and organizations working with Arabic text data, enabling tasks like:

Automated extraction of key information from news archives.
Cross-lingual translation for global accessibility.
Real-time analysis of Arabic news for media monitoring.
Scalable NLP solutions for multilingual datasets.

⚙️ Technical Highlights

Model: Qwen2.5-1.5B-Instruct, fine-tuned with LoRA adapters for efficiency.
Data Handling: Arabic text is tokenized and processed to avoid unwanted tokens (e.g., Chinese characters) using a custom logits processor.
Performance: vLLM ensures low-latency inference, while Locust tests scalability.
Environment: Built for Google Colab with GPU support, but adaptable to local or cloud setups.


⚠️ Note: This is an experimental project for educational purposes. It should not be used in production without rigorous validation, especially for sensitive applications like news analysis or legal contexts.


✨ Features

🧠 Fine-tuned Qwen LLM for Arabic-specific NLP tasks.
📝 Structured JSON output with Pydantic schema validation.
🌐 Interactive Gradio UI for extraction and translation tasks.
⚡ High-performance inference with vLLM and LoRA adapters.
📈 Scalability testing with Locust for concurrent user simulation.
🔧 Integration with Google Colab, WandB, and Hugging Face for seamless workflows.


🛠️ Requirements

🐍 Python 3.10+
🤗 Transformers & Hugging Face Hub
📦 Gradio, vLLM, Locust, Faker, json_repair
🔬 Additional libraries: Torch, WandB, Groq, etc.

See requirements.txt for the complete list.

🚀 Installation

Clone the Repository:
bashgit clone https://github.com/yourusername/qwen-llm-finetuning.git
cd qwen-llm-finetuning

Set Up a Virtual Environment:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
bashpip install -r requirements.txt
Install LLaMA-Factory:
bashgit clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .

Authenticate Services:

Set up WandB and Hugging Face API keys (via environment variables or Colab secrets).




📡 Usage
1. Fine-Tuning the Model
Open FineTuning_LLMs_FromScratch.ipynb in Jupyter or Colab:

Mount Google Drive for data and model storage.
Install dependencies and authenticate with WandB/Hugging Face.
Fine-tune the Qwen model using LLaMA-Factory:
bashllamafactory-cli train --model_name Qwen/Qwen2.5-1.5B-Instruct --dataset your_dataset.json

Save adapters to /gdrive/MyDrive/FineTuning_fromScratch/models.

2. Deploying with Gradio
Open Deploy_Qwen_LLM.ipynb:

Load the fine-tuned model with adapters.
Launch the Gradio interface:
pythoniface.launch(share=True)

Input Arabic text, select a task (Extraction Details or Translation), and view results.
Access the public URL for remote testing (valid for 1 week).

3. Inference with vLLM
Serve the model:
bashvllm serve "Qwen/Qwen2.5-1.5B-Instruct" --dtype=half --gpu-memory-utilization 0.8 --max_lora_rank 64 --enable-lora --lora-modules articls-lora="/gdrive/MyDrive/FineTuning_fromScratch/models"
Test via API:
pythonimport requests
response = requests.post("http://localhost:8000/v1/completions", json={
    "model": "articls-lora",
    "prompt": "Your Arabic prompt here",
    "max_tokens": 1000,
    "temperature": 0.3
})
print(response.json())
4. Load Testing with Locust
Run the load test:
bashlocust --headless -f locust.py --host=http://localhost:8000 -u 20 -r 1 -t "60s" --html=Test_Speed_LLM.html
This simulates 20 users sending random Arabic prompts.

🌐 API Endpoints (vLLM Server)



















EndpointMethodDescriptionInput ExampleOutput Example/v1/completionsPOSTGenerate completions from prompt{"model": "articls-lora", "prompt": "Arabic text", "max_tokens": 1000}{"choices": [{"text": "Generated JSON or translation"}]} 

🧠 Model Tasks & Schemas
Extraction Details

Output: JSON with structured article details.
Schema (Pydantic):

story_title: String (5-300 chars)
story_keywords: List[str] (min 1)
story_summary: List[str] (1-5 items)
story_category: Enum (politics, sports, art, etc.)
story_entities: List[Entity] (1-10 items, with value & type)



Translation

Output: JSON with English translation.
Schema (Pydantic):

translated_title: String (5-300 chars)
translated_content: String (min 5 chars)



Example Input (Arabic Article):
textفي أبريل/نيسان 2022، دخل شخصان مكتبة جامعة تارتو...
Example Output (Extraction):
json{
  "story_title": "The Case of Book Thieves in Europe",
  "story_keywords": ["book theft", "Russian literature", "libraries"],
  "story_summary": ["Two individuals stole rare books from European libraries.", "Europol launched Operation Pushkin."],
  "story_category": "art",
  "story_entities": [
    {"entity_value": "University of Tartu", "entity_type": "organization"},
    {"entity_value": "Alexander Pushkin", "entity_type": "person-male"}
  ]
}

📂 Project Structure
text├── 📓 FineTuning_LLMs_FromScratch.ipynb  # Notebook for fine-tuning
├── 📓 Deploy_Qwen_LLM.ipynb              # Notebook for Gradio deployment
├── 💾 models/                            # Fine-tuned LoRA adapters
├── 📄 locust.py                          # Load testing script
├── 📋 requirements.txt                   # Dependencies
├── 📝 README.md                          # This file
└── 📜 LICENSE                            # MIT License

🤝 Contributing
Contributions are welcome to enhance fine-tuning or add features! 🚀

Fork the repository.
Create a feature branch:
bashgit checkout -b feature/your-new-feature

Commit changes:
bashgit commit -m "Add your new feature"

Push to the branch:
bashgit push origin feature/your-new-feature

Open a pull request.

For major changes, open an issue first.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
