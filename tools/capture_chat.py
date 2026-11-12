"""Capture DrugRAG chat screenshot — v5: JS tab selection."""
import os
import tempfile
import time
from playwright.sync_api import sync_playwright

URL = "https://mshz88-drugrag.hf.space"
OUT = "BHI_2026/figures/F4_dashboard.png"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1200})

    print(f"Navigating to {URL}...")
    page.goto(URL, wait_until="domcontentloaded", timeout=180_000)
    print("Page loaded. Waiting for Gradio...")
    time.sleep(12)

    # Use JS to find and click the Chat tab button
    print("Selecting Chat tab via JS...")
    result = page.evaluate("""
        () => {
            const buttons = document.querySelectorAll('button.svelte-11gaq1');
            for (const btn of buttons) {
                if (btn.textContent.includes('Chat')) {
                    btn.click();
                    return 'clicked: ' + btn.textContent.trim();
                }
            }
            // Try role=tab buttons
            const tabs = document.querySelectorAll('[role="tab"]');
            for (const tab of tabs) {
                if (tab.textContent.includes('Chat')) {
                    tab.click();
                    return 'clicked-tab: ' + tab.textContent.trim();
                }
            }
            return 'not found. buttons=' + buttons.length + ' tabs=' + tabs.length;
        }
    """)
    print(f"JS result: {result}")
    time.sleep(5)

    page.screenshot(path=os.path.join(tempfile.gettempdir(), "drugrag_v5_chat.png"))
    print("Chat tab screenshot taken.")

    # Now find the question input and Run button
    print("Looking for question input...")
    # Gradio textboxes are typically <textarea> or <input>
    input_filled = page.evaluate("""
        () => {
            const ta = document.querySelector('textarea');
            if (ta) { ta.value = 'What is the mechanism of Memantine in angiogenesis?'; ta.dispatchEvent(new Event('input', {bubbles:true})); return 'textarea'; }
            const inputs = document.querySelectorAll('input[type="text"]');
            for (const inp of inputs) {
                if (inp.placeholder && inp.placeholder.toLowerCase().includes('mechanism')) {
                    inp.value = 'What is the mechanism of Memantine in angiogenesis?';
                    inp.dispatchEvent(new Event('input', {bubbles:true}));
                    return 'input: ' + inp.placeholder;
                }
            }
            // Try all text inputs
            for (const inp of inputs) {
                inp.value = 'What is the mechanism of Memantine in angiogenesis?';
                inp.dispatchEvent(new Event('input', {bubbles:true}));
                return 'input-generic';
            }
            return 'no-input-found';
        }
    """)
    print(f"Input: {input_filled}")
    time.sleep(2)

    # Click Run via JS
    print("Clicking Run via JS...")
    run_result = page.evaluate("""
        () => {
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {
                if (btn.textContent.includes('Run')) {
                    btn.click();
                    return 'clicked: ' + btn.textContent.trim().substring(0, 50);
                }
            }
            return 'not found. total buttons=' + buttons.length;
        }
    """)
    print(f"Run: {run_result}")

    print("Waiting 85s for both answers to generate...")
    time.sleep(85)

    # Take screenshots
    print(f"Taking screenshots...")
    page.screenshot(path=OUT, full_page=False)
    page.screenshot(path=OUT.replace(".png", "_full.png"), full_page=True)
    page.screenshot(path=os.path.join(tempfile.gettempdir(), "drugrag_v5_result.png"),
                    full_page=False)
    print(f"Done! Saved {OUT}")

    browser.close()
