import asyncio
import argparse
import logging
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Tuple
import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field, ValidationError

from agents import Agent, Runner, trace, function_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pr_reviewer")

# Define configuration model
class Config(BaseModel):
    github_token: str = Field(..., description="GitHub personal access token with repo scope")
    openai_api_key: str = Field(..., description="OpenAI API key")
    repos_file: str = Field("repositories.txt", description="Path to file containing repositories to monitor")
    interval: int = Field(3600, description="Checking interval in seconds")
    model: str = Field("o4-mini", description="OpenAI model to use")
    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for model response")
    
    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "Config":
        """Load configuration from environment variables and command line arguments."""
        # First, load from .env file if it exists
        if os.path.exists(".env"):
            load_dotenv(override=True)
        
        # Priority: CLI args > environment variables
        config_dict = {
            "github_token": args.token or os.environ.get("GITHUB_TOKEN", ""),
            "openai_api_key": args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            "repos_file": args.repos_file,
            "interval": args.interval,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        
        try:
            return cls(**config_dict)
        except ValidationError as e:
            missing_fields = [error["loc"][0] for error in e.errors() if error["type"] == "value_error.missing"]
            for field in missing_fields:
                logger.error(f"Missing required configuration: {field}")
            
            if "github_token" in missing_fields:
                logger.error("GITHUB_TOKEN is required. Set it in .env file or use --token")
            if "openai_api_key" in missing_fields:
                logger.error("OPENAI_API_KEY is required. Set it in .env file or use --api-key")
            
            raise ValueError("Missing required configuration values") from e


class ReviewStateManager:
    """Manages the state of reviewed PRs, persisting to SQLite."""
    
    def __init__(self, db_path: str = "pr_reviews.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviewed_prs (
            repo_owner TEXT,
            repo_name TEXT,
            pr_number INTEGER,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (repo_owner, repo_name, pr_number)
        )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"Initialized review state database at {self.db_path}")
    
    def mark_as_reviewed(self, owner: str, repo: str, pr_number: int) -> None:
        """Mark a PR as reviewed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO reviewed_prs (repo_owner, repo_name, pr_number) VALUES (?, ?, ?)",
            (owner, repo, pr_number)
        )
        conn.commit()
        conn.close()
    
    def is_reviewed(self, owner: str, repo: str, pr_number: int) -> bool:
        """Check if a PR has been reviewed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM reviewed_prs WHERE repo_owner = ? AND repo_name = ? AND pr_number = ?",
            (owner, repo, pr_number)
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result
    
    def get_all_reviewed_prs(self) -> List[Tuple[str, str, int]]:
        """Get all reviewed PRs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT repo_owner, repo_name, pr_number FROM reviewed_prs")
        result = cursor.fetchall()
        conn.close()
        return result


class GitHubClient:
    """Client for interacting with GitHub API."""
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}" if token else None
        }
        self.base_url = "https://api.github.com"
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make a request to the GitHub API with retry logic."""
        full_url = f"{self.base_url}/{url.lstrip('/')}" if not url.startswith('http') else url
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=full_url,
                    headers=self.headers,
                    **kwargs
                )
                
                # Handle rate limiting
                if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining == 0:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        current_time = time.time()
                        sleep_time = max(reset_time - current_time, 0) + 1
                        logger.warning(f"Rate limited. Waiting {sleep_time:.0f} seconds until reset.")
                        time.sleep(sleep_time)
                        continue  # Retry after waiting
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise
    
    def verify_repository_exists(self, owner: str, repo: str) -> bool:
        """Verify that a repository exists and is accessible."""
        try:
            self._make_request("GET", f"/repos/{owner}/{repo}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error verifying repository {owner}/{repo}: {e}")
            return False
    
    def get_pull_requests(self, owner: str, repo: str, state: str = "open") -> List[Dict]:
        """Get pull requests for a repository."""
        response = self._make_request("GET", f"/repos/{owner}/{repo}/pulls?state={state}")
        return response.json()
    
    def get_pr_comments(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """Get comments on a pull request."""
        response = self._make_request("GET", f"/repos/{owner}/{repo}/issues/{pr_number}/comments")
        return response.json()
    
    def create_pr_comment(self, owner: str, repo: str, pr_number: int, body: str) -> Dict:
        """Create a comment on a pull request."""
        response = self._make_request(
            "POST", 
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body}
        )
        return response.json()
    
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """Get files changed in a pull request."""
        response = self._make_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
        return response.json()


class ReviewEngine:
    """Engine for reviewing PRs using OpenAI."""
    
    def __init__(self, github_client: GitHubClient, config: Config):
        self.github = github_client
        self.config = config
        self.bot_signature = "*This review was generated by an AI assistant"
        self.review_state = ReviewStateManager()
        
        # Create agent with configured model settings
        self.agent = Agent(
            name="PR Reviewer",
            instructions="""You are an expert code reviewer. 
            
Your reviews should be challenging and thought-provoking. Don't simply agree with the changes or give bland feedback.
Instead, question assumptions, highlight potential issues, and suggest alternatives to make the code better.

Be constructively critical while remaining respectful. Your goal is to help improve the code quality 
by challenging the author to think more deeply about their implementation.""",
            tools=[self.list_pull_requests, self.add_pr_comment, self.get_pr_files],
            model=config.model
        )
        
        # Store model parameters for use with Runner
        self.model_temperature = config.temperature
        self.model_max_tokens = config.max_tokens
    
    @function_tool
    def list_pull_requests(self, owner: str, repo: str, state: str) -> List[Dict]:
        """
        List pull requests for a GitHub repository.
        
        Args:
            owner: Repository owner username
            repo: Repository name
            state: State of PRs to get ('open', 'closed', or 'all')
        
        Returns:
            List of pull request objects
        """
        return self.github.get_pull_requests(owner, repo, state)
    
    @function_tool
    def add_pr_comment(self, owner: str, repo: str, issue_number: int, body: str) -> Dict:
        """
        Add a comment to a GitHub PR.
        
        Args:
            owner: Repository owner username
            repo: Repository name
            issue_number: PR/issue number
            body: Comment text
        
        Returns:
            Response from GitHub API
        """
        return self.github.create_pr_comment(owner, repo, issue_number, body)
    
    @function_tool
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """
        Get the list of files changed in a PR.
        
        Args:
            owner: Repository owner username
            repo: Repository name
            pr_number: PR number
        
        Returns:
            List of file change objects
        """
        return self.github.get_pr_files(owner, repo, pr_number)
    
    async def review_pr(self, owner: str, repo: str, pr_number: int) -> str:
        """Generate a review for a pull request."""
        # Check if PR has too many changes to review
        try:
            files = self.github.get_pr_files(owner, repo, pr_number)
            total_changes = sum(f.get("additions", 0) + f.get("deletions", 0) for f in files)
            
            # If the PR is too large, return a message instead of a review
            if total_changes > 1000:  # Arbitrary limit, adjust as needed
                logger.warning(f"PR #{pr_number} has {total_changes} changes, which may be too large for effective review")
                return (
                    f"This PR contains {total_changes} changes, which is quite large for an automated review. "
                    "Consider breaking large PRs into smaller, more focused changes for better feedback. "
                    "I can provide a general review, but might miss important details due to the size."
                )
        except Exception as e:
            logger.error(f"Error checking PR size: {e}")
            # Continue with review anyway
        
        message = f"""Review PR #{pr_number} in the {owner}/{repo} repository. 

Your goal is to provide challenging, thought-provoking feedback that makes the author reconsider assumptions and improve their code. 

Focus on:
1. Identifying potential bugs or edge cases
2. Questioning design decisions and suggesting alternatives
3. Identifying performance issues or scalability concerns
4. Challenging assumptions in the implementation
5. Finding missed test cases or validation

Don't simply agree with the changes. Be constructively critical while remaining respectful.
"""
        logger.info(f"Reviewing PR #{pr_number} for {owner}/{repo}...")
        try:
            result = await Runner.run(
                self.agent,
                message,
                temperature=self.model_temperature,
                max_tokens=self.model_max_tokens
            )
            return result.final_output
        except Exception as e:
            logger.error(f"Error generating review: {e}")
            return f"Error generating review: {str(e)}"
    
    async def should_review_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        """Determine if a PR should be reviewed based on existing comments and database state."""
        # First check our persistent database
        if self.review_state.is_reviewed(owner, repo, pr_number):
            logger.info(f"PR #{pr_number} already marked as reviewed in database")
            return False
        
        # Then check if we've already commented
        try:
            comments = self.github.get_pr_comments(owner, repo, pr_number)
            for comment in comments:
                if self.bot_signature in comment.get("body", ""):
                    logger.info(f"PR #{pr_number} already has a review comment")
                    # Update our database to mark it as reviewed
                    self.review_state.mark_as_reviewed(owner, repo, pr_number)
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking PR comments: {e}")
            # If we can't check comments, err on the side of not re-reviewing
            return False
    
    async def post_review_comment(self, owner: str, repo: str, pr_number: int, review: str) -> None:
        """Post a review comment on a PR."""
        ai_disclaimer = "\n\n*This review was generated by an AI assistant. The AI is programmed to be critical and challenging to encourage improvement.*"
        full_comment = review + ai_disclaimer
        
        try:
            self.github.create_pr_comment(owner=owner, repo=repo, pr_number=pr_number, body=full_comment)
            logger.info(f"Added review comment to PR #{pr_number}")
            
            # Mark as reviewed in our database
            self.review_state.mark_as_reviewed(owner, repo, pr_number)
        except Exception as e:
            logger.error(f"Error posting review comment: {e}")
    
    async def process_pr(self, owner: str, repo: str, pr_number: int, is_draft: bool, monitor_draft_changes: bool = True) -> None:
        """Process a single PR."""
        try:
            # Skip if PR is draft and we're not monitoring draft changes
            if is_draft and not monitor_draft_changes:
                logger.info(f"Skipping draft PR #{pr_number}")
                return
            
            # Check if we should review this PR
            if not await self.should_review_pr(owner, repo, pr_number):
                return
            
            # Generate and post review
            review = await self.review_pr(owner, repo, pr_number)
            await self.post_review_comment(owner, repo, pr_number, review)
            
        except Exception as e:
            logger.error(f"Error processing PR #{pr_number}: {e}")
    
    async def monitor_repo(self, owner: str, repo: str, monitor_draft_changes: bool = True) -> None:
        """Monitor a single repository for PRs that need review."""
        logger.info(f"Checking for PRs in {owner}/{repo}...")
        
        # Verify the repository exists and is accessible
        if not self.github.verify_repository_exists(owner, repo):
            logger.error(f"Repository {owner}/{repo} could not be accessed. Check that:")
            logger.error(f"1. The repository exists (check capitalization)")
            logger.error(f"2. Your token has correct permissions")
            logger.error(f"3. If it's a private repo, your token has access")
            return
        
        try:
            # Get open PRs
            open_prs = self.github.get_pull_requests(owner=owner, repo=repo, state="open")
            logger.info(f"Found {len(open_prs)} open PRs in {owner}/{repo}")
            
            # Process each PR
            for pr in open_prs:
                pr_number = pr.get("number")
                is_draft = pr.get("draft", False)
                await self.process_pr(owner, repo, pr_number, is_draft, monitor_draft_changes)
                
        except Exception as e:
            logger.error(f"Error monitoring {owner}/{repo}: {e}")
    
    async def monitor_all_repos(self, repo_list: List[str], interval: int, monitor_draft_changes: bool = True) -> None:
        """Monitor multiple repositories for PRs."""
        while True:
            logger.info(f"Starting review cycle for {len(repo_list)} repositories")
            
            # Process each repository
            for repo_path in repo_list:
                try:
                    owner, repo = repo_path.strip().split('/')
                    await self.monitor_repo(owner, repo, monitor_draft_changes)
                except ValueError:
                    logger.error(f"Invalid repository format: {repo_path}. Expected format: owner/repo")
                except Exception as e:
                    logger.error(f"Error monitoring repository {repo_path}: {e}")
            
            logger.info(f"Finished checking all repositories. Waiting {interval} seconds before next check...")
            await asyncio.sleep(interval)


# Configuration functions
def read_repo_list(filepath: str = "repositories.txt") -> List[str]:
    """Read the list of repositories to monitor from a file."""
    repo_list = []
    
    # Create default file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write("# Add repositories to monitor, one per line in format: owner/repo\n")
            f.write("AndreasInk/PR-Review-Agent\n")
        logger.info(f"Created default repositories file at {filepath}")
    
    try:
        with open(filepath, "r") as f:
            for line in f:
                # Skip empty lines and comments
                line = line.strip()
                if line and not line.startswith("#"):
                    repo_list.append(line)
    except Exception as e:
        logger.error(f"Error reading repository list: {e}")
        # Default to the main repository if there's an error
        repo_list = ["AndreasInk/PR-Review-Agent"]
    
    return repo_list


async def run_legacy(directory_path: str):
    """Legacy function for running git repo analysis (kept for reference).
    
    This mode analyzes a local git repository to identify contributors and summarize changes,
    rather than reviewing PRs. It's maintained for backward compatibility and local analysis.
    """
    # For legacy mode, we'll use the basic agent without specific GitHub tools
    agent = Agent(
        name="Assistant",
        instructions=f"Answer questions about the git repository at {directory_path}, use that for repo_path"
    )

    # Default parameters for legacy mode
    temperature = 0.7
    max_tokens = None

    message = "Who's the most frequent contributor?"
    logger.info("\n" + "-" * 40)
    logger.info(f"Running: {message}")
    result = await Runner.run(agent, message, temperature=temperature, max_tokens=max_tokens)
    logger.info(result.final_output)

    message = "Summarize the last change in the repository."
    logger.info("\n" + "-" * 40)
    logger.info(f"Running: {message}")
    result = await Runner.run(agent, message, temperature=temperature, max_tokens=max_tokens)
    logger.info(result.final_output)


def create_env_example():
    """Create a .env.example file if it doesn't exist."""
    if not os.path.exists(".env.example"):
        with open(".env.example", "w") as f:
            f.write("# GitHub API token with repo scope\n")
            f.write("GITHUB_TOKEN=your_github_token_here\n\n")
            f.write("# OpenAI API key\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        logger.info("Created .env.example file")


async def main():
    # Create .env.example if it doesn't exist
    create_env_example()
    
    parser = argparse.ArgumentParser(description="GitHub PR Review Agent")
    parser.add_argument("--repos-file", help="Path to file containing repositories to monitor", default="repositories.txt")
    parser.add_argument("--interval", type=int, default=3600, help="Checking interval in seconds (default: 3600)")
    parser.add_argument("--token", help="GitHub token (can also be set via GITHUB_TOKEN env var)")
    parser.add_argument("--api-key", help="OpenAI API key (can also be set via OPENAI_API_KEY env var)")
    parser.add_argument("--model", help="OpenAI model to use", default="o4-mini")
    parser.add_argument("--temperature", help="Model temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", help="Maximum tokens for model response", type=int, default=None)
    parser.add_argument("--legacy", action="store_true", help="Run legacy git analysis mode for local repositories")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = Config.from_env_and_args(args)
        
        # Set OpenAI API key for the agent to use
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        
        if args.legacy:
            # Legacy mode for local git repository analysis
            directory_path = input("Please enter the path to the git repository: ")
            with trace(workflow_name="Git Repository Analysis"):
                await run_legacy(directory_path)
        else:
            # PR review mode for multiple repositories
            # Read the repositories list
            repos = read_repo_list(config.repos_file)
            if not repos:
                logger.error(f"No repositories found in {config.repos_file}. Add repositories in the format 'owner/repo', one per line.")
                return
            
            # Initialize clients and review engine
            github_client = GitHubClient(config.github_token)
            review_engine = ReviewEngine(github_client, config)
            
            logger.info(f"Starting PR review monitor for {len(repos)} repositories:")
            for repo in repos:
                logger.info(f"  - {repo}")
            logger.info(f"Using model: {config.model}")
            logger.info(f"Checking every {config.interval} seconds")
            
            with trace(workflow_name="GitHub PR Review Monitor"):
                await review_engine.monitor_all_repos(repos, config.interval)
    
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())