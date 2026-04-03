# taskgen

A fast, concurrent SFT (Supervised Fine-Tuning) task generator for distillation datasets. Generates diverse, difficulty-weighted prompts across math, coding, science, computer science, creative writing, and conversational domains — via any OpenAI-compatible API.

## Features

- 45+ domains, 200+ subdomains across 6 categories
- Weighted difficulty sampling (1–10 scale)
- Configurable category distribution
- Concurrent generation with N workers
- OpenAI-compatible API (works with OpenAI, Together, Mistral, local vLLM, etc.)
- JSONL output with metadata per task
- Optional budget cap with per-token cost tracking
- Append mode to resume interrupted runs
- Auto-generated dataset README on completion

## Install

```bash
git clone https://github.com/empero-org/taskgen.git
cd taskgen
cargo build --release
```

Binary will be at `target/release/taskgen`.

## Usage

```bash
taskgen [OPTIONS]
```

### Required

| Flag | Env | Description |
|---|---|---|
| `--api-key <KEY>` | `OPENAI_API_KEY` | API key for the target provider |

### Options

| Flag | Default | Description |
|---|---|---|
| `--api-base <URL>` | `https://api.openai.com/v1` | API base URL |
| `-m, --model <MODEL>` | `gpt-4o-mini` | Model to use |
| `-c, --count <N>` | `250` | Number of tasks to generate |
| `-w, --workers <N>` | `5` | Concurrent workers |
| `-o, --output <FILE>` | `output.jsonl` | Output file path |
| `-t, --temperature <F>` | `0.9` | Sampling temperature |
| `--append` | — | Append to existing output file |
| `--distribution <STR>` | balanced | Category weights (see below) |
| `--difficulty <STR>` | bell curve | Difficulty weights (see below) |
| `--input-price <F>` | — | Input token price per 1M tokens (for cost tracking) |
| `--output-price <F>` | — | Output token price per 1M tokens |
| `--budget <F>` | — | Hard cost cap in USD (requires price flags) |
| `--system-prompt <STR>` | built-in | Override the system prompt |

## Examples

**Basic — generate 500 tasks with GPT-4o-mini:**
```bash
taskgen --api-key $OPENAI_API_KEY -c 500
```

**Local vLLM / Ollama:**
```bash
taskgen --api-base http://localhost:8000/v1 --api-key none -m mistral-7b-instruct -c 1000 -w 10
```

**Together AI with cost tracking and budget cap:**
```bash
taskgen \
  --api-base https://api.together.xyz/v1 \
  --api-key $TOGETHER_API_KEY \
  -m meta-llama/Llama-3-8b-chat-hf \
  -c 2000 -w 20 \
  --input-price 0.20 --output-price 0.20 \
  --budget 1.00
```

**Custom distribution — 50% coding, 30% math, 20% science:**
```bash
taskgen --api-key $OPENAI_API_KEY --distribution "coding=0.5,math=0.3,science=0.2" -c 500
```

**Custom difficulty — only hard tasks (levels 7–10):**
```bash
taskgen --api-key $OPENAI_API_KEY --difficulty "7=0.25,8=0.25,9=0.25,10=0.25" -c 500
```

**Append mode — resume a previous run:**
```bash
taskgen --api-key $OPENAI_API_KEY -c 1000 --append -o my_dataset.jsonl
```

## Output Format

Each line in the JSONL file is a self-contained task record:

```json
{
  "prompt": "Prove that the sum of two odd integers is always even.",
  "domain": "math::Number Theory",
  "subdomain": "primes",
  "difficulty": 4,
  "taskgen_model": "gpt-4o-mini",
  "temperature": 0.9
}
```

A `README.md` summarising run parameters, token usage, and cost is written alongside the output file on completion.

## Domains

| Category | Domains |
|---|---|
| `math` | Algebra, Calculus, Probability, Statistics, Geometry, Number Theory, Discrete Math, Linear Algebra |
| `coding` | Python, Rust, Go, JavaScript, C, C++, C#, Java, Ruby, Lua, SQL, Web Development |
| `science` | Physics, Chemistry, Biology, Earth Science, Astronomy |
| `cs` | Algorithms, Data Structures, OS, Networking, Databases, Compilers, Distributed Systems, ML, Cybersecurity, Software Engineering |
| `creative` | Fiction, Poetry, Screenwriting, Journalism, Songwriting, Game Narrative, Copywriting, Blogging |
| `conversation` | Debate, Advice, Interview, Teaching, Roleplay |

## Difficulty Scale

| Level | Label |
|---|---|
| 1 | Very Easy (child-level) |
| 2 | Easy (elementary) |
| 3 | Basic (middle school) |
| 4 | Intermediate (high school) |
| 5 | Standard (undergraduate intro) |
| 6 | Skilled (undergraduate advanced) |
| 7 | Proficient (graduate level) |
| 8 | Advanced (professional / researcher) |
| 9 | Expert (top specialist) |
| 10 | Polymath (1-in-a-million genius) |

## Support

If this tool has been useful, consider supporting the project:

- **BTC**: `bc1qx6zepu6sfkvshgdmc4ewu6pk6rpadvpgffpp7v`
- **LTC**: `ltc1qv2mefzps2vtjcpwfx8xxdrpplrcvltswm68r7x`
- **XMR**: `42Dbm5xg5Nq26fdyzfEU7KBnAJfhi7Cvz5J2ex5CzHXkfKuNEJzYCcmJ1GTbgjFZ5MBx72sdG1G9239Cd6rsZfv4QeDkYJY`

---

*by [empero-ai](https://github.com/empero-org)*
