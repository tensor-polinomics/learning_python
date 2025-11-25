#!/usr/bin/env python3
"""
Script to scrape a Medium article with authentication.
Requires playwright to be installed: pip install playwright
After installing, run: playwright install chromium
"""

import asyncio
import json
from playwright.async_api import async_playwright
import sys


async def scrape_medium_article(url, cookies_file=None):
    """
    Scrape a Medium article using Playwright.

    Args:
        url: The Medium article URL
        cookies_file: Optional path to a JSON file containing cookies
    """
    async with async_playwright() as p:
        # Launch browser (headless=False so you can see what's happening)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Load cookies if provided
        if cookies_file:
            try:
                with open(cookies_file, 'r') as f:
                    cookies = json.load(f)
                    await context.add_cookies(cookies)
                print(f"Loaded cookies from {cookies_file}")
            except Exception as e:
                print(f"Warning: Could not load cookies: {e}")

        page = await context.new_page()

        # Navigate to the article
        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="networkidle")

        # Wait a bit for the page to fully render
        await page.wait_for_timeout(3000)

        # Try to extract the article content
        try:
            # Wait for article to load
            await page.wait_for_selector('article', timeout=10000)

            # Extract title
            title = await page.locator('h1').first.text_content()

            # Extract author
            author = ""
            try:
                author = await page.locator('[data-testid="authorName"]').first.text_content()
            except:
                pass

            # Extract date
            date = ""
            try:
                date = await page.locator('time').first.get_attribute('datetime')
            except:
                pass

            # Extract article content
            # Medium articles are typically in an <article> tag
            article_element = page.locator('article').first

            # Get all paragraphs, headings, code blocks, etc.
            content_elements = await article_element.locator('p, h1, h2, h3, h4, h5, h6, pre, ul, ol, blockquote, figure').all()

            content_parts = []

            for element in content_elements:
                tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                text = await element.text_content()

                if not text or not text.strip():
                    continue

                if tag_name == 'h1':
                    content_parts.append(f"# {text.strip()}\n")
                elif tag_name == 'h2':
                    content_parts.append(f"## {text.strip()}\n")
                elif tag_name == 'h3':
                    content_parts.append(f"### {text.strip()}\n")
                elif tag_name == 'h4':
                    content_parts.append(f"#### {text.strip()}\n")
                elif tag_name == 'h5':
                    content_parts.append(f"##### {text.strip()}\n")
                elif tag_name == 'h6':
                    content_parts.append(f"###### {text.strip()}\n")
                elif tag_name == 'pre':
                    content_parts.append(f"```\n{text.strip()}\n```\n")
                elif tag_name == 'blockquote':
                    content_parts.append(f"> {text.strip()}\n")
                elif tag_name in ['ul', 'ol']:
                    # Get list items
                    items = await element.locator('li').all()
                    for item in items:
                        item_text = await item.text_content()
                        content_parts.append(f"- {item_text.strip()}\n")
                else:  # paragraph
                    content_parts.append(f"{text.strip()}\n")

            # Combine everything
            markdown_content = f"# {title}\n\n"
            if author:
                markdown_content += f"**Author:** {author}\n\n"
            if date:
                markdown_content += f"**Date:** {date}\n\n"
            markdown_content += f"**Source:** {url}\n\n"
            markdown_content += "---\n\n"
            markdown_content += "\n".join(content_parts)

            # Save to file
            output_file = "./reports/chronos.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"\n✓ Article scraped successfully!")
            print(f"✓ Saved to {output_file}")
            print(f"\nTitle: {title}")
            print(f"Content length: {len(markdown_content)} characters")

        except Exception as e:
            print(f"Error extracting article: {e}")
            print("\nTrying alternative method...")

            # Fallback: just get all text from article
            try:
                article_text = await page.locator('article').first.text_content()
                with open("./reports/chronos.md", 'w', encoding='utf-8') as f:
                    f.write(f"# Medium Article\n\n**Source:** {url}\n\n---\n\n{article_text}")
                print("✓ Saved article using fallback method")
            except:
                print("✗ Could not extract article content")

        # Keep browser open for 5 seconds so you can see the result
        await page.wait_for_timeout(5000)

        await browser.close()


async def save_cookies_interactive():
    """
    Launch browser for user to log in, then save cookies for future use.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        print("\n" + "="*60)
        print("BROWSER OPENED - Please log in to Medium")
        print("="*60)
        print("\n1. Log in to your Medium account in the browser window")
        print("2. Press ENTER here when you're logged in...")

        await page.goto("https://medium.com")

        # Wait for user to press Enter
        input()

        # Save cookies
        cookies = await context.cookies()
        with open('medium_cookies.json', 'w') as f:
            json.dump(cookies, f, indent=2)

        print("\n✓ Cookies saved to medium_cookies.json")
        print("You can now run the script again with these cookies!\n")

        await browser.close()


if __name__ == "__main__":
    article_url = "https://medium.com/data-science-collective/chronos-2-cold-start-forecasting-with-short-histories-and-no-training-a-practical-tutorial-fbc9dea96278"

    if len(sys.argv) > 1 and sys.argv[1] == "--save-cookies":
        print("Starting cookie save mode...")
        asyncio.run(save_cookies_interactive())
    else:
        # Check if cookies file exists
        import os
        cookies_file = "medium_cookies.json" if os.path.exists("medium_cookies.json") else None

        if not cookies_file:
            print("\nNo cookies file found. You have two options:")
            print("\n1. Run: python scrape_medium.py --save-cookies")
            print("   (This will let you log in and save your session)")
            print("\n2. Run the script anyway and log in manually in the browser")
            print("\nProceed anyway? (y/n): ", end="")

            choice = input().lower()
            if choice != 'y':
                sys.exit(0)

        asyncio.run(scrape_medium_article(article_url, cookies_file))
