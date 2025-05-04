import asyncio
import argparse
import time
from typing import List, Dict, Optional, Set
import os
import requests
import json
from pathlib import Path

from agents import Agent, Runner, trace, function_tool


# Configuration functions
def read_repo_list(filepath: str = "repositories.txt") -> List[str]:
    """Read the list of repositories to monitor from a file."""
    repo_list = []
    
    # Create default file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write("# Add repositories to monitor, one per line in format: owner/repo\n")
            f.write("AndreasInk/NIV2IMV-predictor\n")
        print(f"Created default repositories file at {filepath}")
    
    try:
        with open(filepath, "r") as f:
            for line in f:
                # Skip empty lines and comments
                line = line.strip()
                if line and not line.startswith("#"):
                    repo_list.append(line)
    except Exception as e:
        print(f"Error reading repository list: {e}")
        # Default to the main repository if there's an error
        repo_list = ["AndreasInk/NIV2IMV-predictor"]
    
    return repo_list


# Regular functions for direct calls
def get_pull_requests(owner: str, repo: str, state: str = "open") -> List[Dict]:
    """List open pull requests for a GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state={state}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    # Add authorization if GitHub token is available
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


def create_pr_comment(owner: str, repo: str, issue_number: int, body: str) -> Dict:
    """Add a comment to a GitHub PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    # Add authorization if GitHub token is available
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    else:
        raise ValueError("GITHUB_TOKEN environment variable must be set to comment on PRs")
    
    data = {"body": body}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()


def get_pr_comments(owner: str, repo: str, issue_number: int) -> List[Dict]:
    """Get all comments on a GitHub PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    # Add authorization if GitHub token is available
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


# Tools for the agent to use
@function_tool
def list_pull_requests(owner: str, repo: str, state: str) -> List[Dict]:
    """
    List pull requests for a GitHub repository.
    
    Args:
        owner: Repository owner username
        repo: Repository name
        state: State of PRs to get ('open', 'closed', or 'all')
    
    Returns:
        List of pull request objects
    """
    return get_pull_requests(owner, repo, state)


@function_tool
def add_pr_comment(owner: str, repo: str, issue_number: int, body: str) -> Dict:
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
    return create_pr_comment(owner, repo, issue_number, body)


@function_tool
def get_pr_files(owner: str, repo: str, pr_number: int) -> List[Dict]:
    """
    Get the list of files changed in a PR.
    
    Args:
        owner: Repository owner username
        repo: Repository name
        pr_number: PR number
    
    Returns:
        List of file change objects
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    # Add authorization if GitHub token is available
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()


async def review_pr(agent: Agent, owner: str, repo: str, pr_number: int) -> str:
    """Review a specific PR and return feedback"""
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
    print(f"\nReviewing PR #{pr_number} for {owner}/{repo}...")
    result = await Runner.run(agent, message)
    return result.final_output


async def post_review_comment(owner: str, repo: str, pr_number: int, review: str) -> None:
    """Add a comment to a PR with the review feedback"""
    ai_disclaimer = "\n\n*This review was generated by an AI assistant. The AI is programmed to be critical and challenging to encourage improvement.*"
    full_comment = review + ai_disclaimer
    
    result = create_pr_comment(owner=owner, repo=repo, issue_number=pr_number, body=full_comment)
    print(f"Added review comment to PR #{pr_number}")


async def should_review_pr(owner: str, repo: str, pr_number: int, bot_signature: str) -> bool:
    """Determine if we should review a PR by checking if we've already commented"""
    try:
        comments = get_pr_comments(owner, repo, pr_number)
        for comment in comments:
            if bot_signature in comment.get("body", ""):
                print(f"PR #{pr_number} already has a review comment")
                return False
        return True
    except Exception as e:
        print(f"Error checking PR comments: {e}")
        # If we can't check comments, err on the side of providing a review
        return True


async def monitor_repo(agent: Agent, owner: str, repo: str, reviewed_prs: Set[int], monitor_draft_changes: bool = True):
    """Monitor a single repository for PRs"""
    print(f"\nChecking for PRs in {owner}/{repo}...")
    try:
        # Use the regular function for direct API calls
        open_prs = get_pull_requests(owner=owner, repo=repo, state="open")
        
        for pr in open_prs:
            pr_number = pr.get("number")
            is_draft = pr.get("draft", False)
            
            # Create a unique ID for this PR that includes the repo info
            pr_key = f"{owner}/{repo}#{pr_number}"
            
            # Skip if PR is still in draft mode and we're not checking for changes
            if is_draft and not monitor_draft_changes:
                continue
            
            # Skip if we've already reviewed this PR
            if pr_key in reviewed_prs:
                continue
            
            # Check if it's in draft mode but we should still proceed
            if is_draft and not monitor_draft_changes:
                continue
            
            # Check if we've already commented on this PR
            if not await should_review_pr(owner, repo, pr_number, BOT_SIGNATURE):
                # Mark as reviewed to avoid checking again
                reviewed_prs.add(pr_key)
                continue
            
            # Generate review
            review = await review_pr(agent, owner, repo, pr_number)
            
            # Add comment to PR
            await post_review_comment(owner, repo, pr_number, review)
            
            # Mark as reviewed
            reviewed_prs.add(pr_key)
    
    except Exception as e:
        print(f"Error checking {owner}/{repo}: {e}")


async def monitor_all_repos(repo_list: List[str], interval: int, reviewed_prs: Set[str], monitor_draft_changes: bool = True):
    """Monitor multiple repositories for PRs"""
    # Create an agent with GitHub review tools - we'll use one agent for all repositories
    agent = Agent(
        name="PR Reviewer",
        instructions="""You are an expert code reviewer. 
        
Your reviews should be challenging and thought-provoking. Don't simply agree with the changes or give bland feedback.
Instead, question assumptions, highlight potential issues, and suggest alternatives to make the code better.

Be constructively critical while remaining respectful. Your goal is to help improve the code quality 
by challenging the author to think more deeply about their implementation.""",
        tools=[list_pull_requests, add_pr_comment, get_pr_files],
        model="o4-mini"  # Use o4-mini model
    )

    while True:
        # Process each repository in turn
        for repo_path in repo_list:
            try:
                owner, repo = repo_path.strip().split('/')
                await monitor_repo(agent, owner, repo, reviewed_prs, monitor_draft_changes)
            except ValueError:
                print(f"Invalid repository format: {repo_path}. Expected format: owner/repo")
            except Exception as e:
                print(f"Error monitoring repository {repo_path}: {e}")
        
        # Wait for the next check
        print(f"\nFinished checking all repositories. Waiting {interval} seconds before next check...")
        await asyncio.sleep(interval)


async def run_legacy(directory_path: str):
    """Legacy function for running git repo analysis (kept for reference)"""
    # For legacy mode, we'll use the basic agent without specific GitHub tools
    agent = Agent(
        name="Assistant",
        instructions=f"Answer questions about the git repository at {directory_path}, use that for repo_path"
    )

    message = "Who's the most frequent contributor?"
    print("\n" + "-" * 40)
    print(f"Running: {message}")
    result = await Runner.run(agent, message)
    print(result.final_output)

    message = "Summarize the last change in the repository."
    print("\n" + "-" * 40)
    print(f"Running: {message}")
    result = await Runner.run(agent, message)
    print(result.final_output)


# Global constants
BOT_SIGNATURE = "*This review was generated by an AI assistant"


async def main():
    parser = argparse.ArgumentParser(description="GitHub PR Review Agent")
    parser.add_argument("--repos-file", help="Path to file containing repositories to monitor", default="repositories.txt")
    parser.add_argument("--interval", type=int, default=3600, help="Checking interval in seconds (default: 3600)")
    parser.add_argument("--token", help="GitHub token (can also be set via GITHUB_TOKEN env var)")
    parser.add_argument("--legacy", action="store_true", help="Run legacy git analysis mode")
    args = parser.parse_args()

    # Set GitHub token from args if provided
    if args.token:
        os.environ["GITHUB_TOKEN"] = args.token

    if args.legacy:
        # Legacy mode
        directory_path = input("Please enter the path to the git repository: ")
        with trace(workflow_name="Git Repository Analysis"):
            await run_legacy(directory_path)
    else:
        # PR review mode for multiple repositories        
        # Check if GitHub token is set
        if not os.environ.get("GITHUB_TOKEN"):
            token = input("Enter your GitHub token (required to comment on PRs): ")
            if token:
                os.environ["GITHUB_TOKEN"] = token
            else:
                print("Warning: No GitHub token provided. You may not be able to comment on PRs.")
        
        # Read the repositories list
        repos = read_repo_list(args.repos_file)
        if not repos:
            print(f"No repositories found in {args.repos_file}. Add repositories in the format 'owner/repo', one per line.")
            return
            
        print(f"Starting PR review monitor for {len(repos)} repositories:")
        for repo in repos:
            print(f"  - {repo}")
        print(f"Checking every {args.interval} seconds")
        
        with trace(workflow_name="GitHub PR Review Monitor"):
            # Set to track which PRs have been reviewed (using full repo/number identifiers)
            reviewed_prs = set()
            await monitor_all_repos(repos, args.interval, reviewed_prs)


if __name__ == "__main__":
    asyncio.run(main())