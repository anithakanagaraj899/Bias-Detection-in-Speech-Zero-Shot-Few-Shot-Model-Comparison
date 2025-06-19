# Bias-Detection-in-Speech-Zero-Shot-Few-Shot-Model-Comparison
This project addresses the critical challenge of detecting bias in speech, particularly in real-world scenarios such as social media posts, political speeches, and recorded conversations. Since obtaining large, fully labeled datasets is challenging, regular machine learning methods can struggle. To determine which approach works best with limited labeled data, we explored and compared various methods for detecting bias, including zero-shot learning, few-shot learning, traditional machine learning, and transformer-based models.
To make the process more realistic, we utilized Whisper AI to transcribe speech recordings into text, then cleaned and prepared the data for further analysis. We used the ToxicBias dataset, as described in a research paper by Sahoo et al. (2022), which is based on the Jigsaw Toxicity dataset and incorporates additional labels that indicate the type of bias and its reasoning. This helped us understand various kinds of bias, including political, gender, racial, religious, and LGBTQ-related biases.
Ultimately, we compared all the methods to understand which ones work well, their limitations, and their usefulness in real-world bias detection.

## Objective

To develop and compare machine learning methods for speech-based bias detection using the Whisper AI transcription tool and the ToxicBias dataset. The focus is on achieving strong performance with minimal labeled data.

## Dataset

- **ToxicBias Dataset** (based on Jigsaw Toxicity Dataset)
  - Includes toxicity labels along with fine-grained bias labels and reasoning.
  - Used for supervised and zero-/few-shot experiments.
- **Speech Data**
  - Transcribed using OpenAI's Whisper to simulate real-world applications.

## Methodology

### Preprocessing
- Audio-to-text transcription with Whisper AI
- Text cleaning and formatting
- Tokenization and vectorization

### Models Compared
- **Traditional ML**: Logistic Regression, SVM, Random Forest
- **Transformer Models**: BERT, RoBERTa (Fine-tuned)
- **Zero-shot Learning**: Prompt-based classification with pretrained language models
- **Few-shot Learning**: Leveraging small labeled subsets to train LLMs

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

## Results & Analysis

- Compared model performance across various metrics
- Observed trade-offs in accuracy vs data requirements
- Zero-shot models performed surprisingly well on some bias types
- Few-shot fine-tuned models offered the best balance of performance and adaptability

## Key Findings

- Whisper AI enabled practical bias detection from raw speech
- Zero-shot and few-shot learning are valuable when labeled data is scarce
- Transformer-based models outperformed traditional methods for nuanced bias detection

## Tools & Libraries

- Python, NumPy, Pandas, scikit-learn
- Hugging Face Transformers
- OpenAI Whisper
- Matplotlib, Seaborn for visualization
- Prompt Engineering: Zero-shot learning, Few-shot learning
- ## How to Run
1. Clone the repo  
2. Install dependencies: pip install -r requirements.tx 
3. Run the notebook in notebooks/ to see preprocessing, modeling, and results  

------------------------------------------------------------------------------------------------------------

Note: This project is for educational purposes.
