# PR-Review-Agent

An automated GitHub PR review agent powered by OpenAI that provides thoughtful, critical code reviews across multiple repositories.

![Screenshot 2025-05-04 at 7 39 25 PM](https://github.com/user-attachments/assets/f4f69bf6-9d36-4067-a117-dc72cd0ae85b)

[An example of the agent's feedback](https://github.com/AndreasInk/PR-Review-Agent/pull/1#issuecomment-2849487178)

## Features

- 🤖 AI-powered code reviews that challenge assumptions and improve code quality
- 📊 Monitor multiple GitHub repositories for new PRs
- 🔄 Configurable checking intervals
- ⚙️ Avoids duplicate comments on PRs
- 📝 Critical but constructive feedback
- 🔍 Reviews code with the OpenAI o4-mini model

## Installation

### Prerequisites

- Python 3.8 or higher
- A GitHub personal access token with repo permissions
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PR-Review-Agent.git
   cd PR-Review-Agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   GITHUB_TOKEN=your_github_token
   OPENAI_API_KEY=your_openai_api_key
   ```

## Configuration

### Repository List

The agent monitors repositories listed in a `repositories.txt` file in the project root. Format:

```
# Add repositories to monitor, one per line
username/repo1
username/repo2
```

If the file doesn't exist, it will be created automatically with default settings.

### GitHub Token

You need a GitHub personal access token with `repo` scope to enable the agent to read PRs and post comments.

To create a token:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope
3. Add it to your `.env` file or provide it when running the agent

## Usage

### Basic Usage

Run the agent with default settings:

```bash
python main.py
```

This will:
- Check repositories listed in `repositories.txt`
- Use default 1-hour interval between checks
- Look for GitHub token in `.env` file or prompt you to enter one

### Command Line Options

```bash
python main.py --repos-file custom_repos.txt --interval 1800 --token your_github_token
```

Options:
- `--repos-file`: Path to a custom repository list file (default: `repositories.txt`)
- `--interval`: Checking interval in seconds (default: 3600)
- `--token`: GitHub token (can also be set via GITHUB_TOKEN env var)
- `--legacy`: Run legacy git analysis mode

### Customizing PR Reviews

To customize the type of feedback provided by the agent, modify the review prompts in the `review_pr` function in `main.py`.

## How It Works

1. The agent reads the repository list from the configuration file
2. For each repository, it fetches open PRs via the GitHub API
3. For each PR that hasn't been reviewed yet, it:
   - Uses OpenAI's o4-mini model to generate a code review
   - Posts the review as a comment on the PR
   - Tracks the PR to avoid duplicate comments
4. Waits for the configured interval, then repeats

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request

### Development Guidelines

- Use descriptive commit messages
- Add tests for new features
- Update documentation as needed
- Follow the existing code style

## License

[MIT License](LICENSE)

## Acknowledgements

- This project uses the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- Inspired by the need for automated, critical code reviews that challenge developers to improve their code
