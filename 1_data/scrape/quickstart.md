# Quick Start Guide - NeurIPS 2025 Paper Scraper

## 🚀 Getting Started (3 steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Test Your Setup

```bash
python test_connection.py
```

You should see:
```
✅ Successfully connected to OpenReview API
✅ Successfully fetched 5 sample papers
✅ All tests passed! You're ready to scrape.
```

### Step 3: Run the Scraper

**Option A: Basic Scraper (Simple)**
```bash
python scrape_neurips_2025.py
```

**Option B: Enhanced Scraper (With Filters)**
```bash
python scrape_neurips_2025_enhanced.py
```

## 📊 What You Get

After running, you'll have:
- `neurips_2025_papers.json` - All **accepted** papers in JSON format
- `neurips_2025_papers.csv` - All **accepted** papers in CSV format (easy to open in Excel)

**Important**: Only accepted papers are included (~5,780 papers). Rejected submissions are automatically filtered out! 🎯

## 🎯 Common Use Cases

### Get Only Oral Papers
```bash
python scrape_neurips_2025_enhanced.py --award Oral -o orals.json
```

### Get Only Main Track Papers
```bash
python scrape_neurips_2025_enhanced.py --tracks Main -o main_track.json
```

### Get Spotlights from Both Tracks
```bash
python scrape_neurips_2025_enhanced.py --award Spotlight -o spotlights.json
```

## ⏱️ How Long Does It Take?

- **Full scrape** (~5,780 accepted papers): ~2 minute
- **Single track**: < 1 minute
- **Filtered by award**: < 1 minute (depending on filter)

The scraper shows a progress bar so you know how it's going!

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'openreview'"
```bash
pip install openreview-py
```

### Connection errors or timeouts
- Check your internet connection
- The OpenReview API might be busy, try again in a few minutes
- If it keeps failing, try adding delays (script already has some built in)

### No papers returned
- Make sure you're using the correct track names: `Main` or `Datasets and Benchmarks`
- Check that NeurIPS 2025 papers are actually published on OpenReview

## 💡 Pro Tips

1. **Start with test_connection.py** - Always verify your setup first
2. **Use filters** - If you only need specific papers, use the enhanced scraper with filters to save time
3. **CSV for quick viewing** - The CSV file can be opened in Excel/Google Sheets for easy browsing
4. **JSON for programming** - Use the JSON file if you're building a visualizer or other tool

## 📝 Example Output Structure

```json
{
  "paper": "Scaling Laws for Reward Model Overoptimization",
  "authors": ["Leo Gao", "John Schulman", "Jacob Hilton"],
  "abstract": "In reinforcement learning from human feedback...",
  "link": "https://openreview.net/forum?id=abc123",
  "award": "Oral",
  "track": "Main"
}
```

## 🎨 Next Steps

After scraping, you can:
1. Load the JSON/CSV into your visualizer
2. Filter papers by keywords in abstracts
3. Analyze author collaborations
4. Create citation networks
5. Build an interactive paper browser

Good luck with your NeurIPS 2025 visualizer! 🎉