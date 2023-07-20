# mortGPT

![mortGPT](https://github.com/EveryOneIsGross/mortGPT/assets/23621140/bff2583b-b244-43eb-a68f-a50e636f1457)

## Overview

mortGPT is a unique project that brings the concept of mortality to chatbots. It's a playful and innovative approach to AI conversation models, introducing the idea of time and lifespan into the chatbot's memory and responses. This project is built on OpenAI's GPT-3 and uses advanced natural language processing techniques to create a more human-like interaction experience. Currently is a very basic implementation for the bot to be aware of time and the passing of a "day". 

## Features

- **Time Awareness**: mortGPT has a concept of time, represented in token-seconds. The chatbot's "day" progresses as it interacts with users, adding a new dimension to the conversation.

- **Memory Management**: The chatbot maintains a memory of past interactions, allowing it to reference previous topics and provide more context-aware responses.

- **Sentiment Analysis**: Using the NLTK Sentiment Intensity Analyzer, mortGPT can understand the sentiment of user inputs and respond accordingly.

- **Keyword Extraction**: The chatbot uses the Rake algorithm to extract keywords from user inputs, aiding in topic detection and response generation.

- **Cosine Similarity**: mortGPT calculates the cosine similarity between the user's time and the chatbot's time, adding another layer of context to the conversation.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/mortGPT.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
4. Run the script:
   ```
   python main.py
   ```

## Usage

Interact with the chatbot through the command line interface. The chatbot will respond to your inputs based on its current "time" and the topics you discuss.

## Contributing

We welcome contributions to mortGPT! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Contact

If you have any questions or feedback, please feel free to reach out to us. You can also [open an issue](https://github.com/username/mortGPT/issues) on this GitHub repository.

Enjoy your conversations with mortGPT!
