use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use chrono::Local;
use clap::Parser;
use futures::stream::{self, StreamExt};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are a task generator for SFT (Supervised Fine-Tuning) distillation datasets. Given a domain, subdomain, and difficulty level, generate a single, self-contained prompt/task that a language model would be expected to respond to.

Rules:
- The task MUST be directly and specifically about the given subdomain. The subdomain must be the central focus of the task, not just the broader domain category.
- The difficulty must match the requested level (1-10 scale where 1=basic child-level, 10=polymath/genius expert).
- Output ONLY the task prompt itself, nothing else. No preamble, no explanation, no labels.
- The task should be realistic and useful for training purposes.
- For coding tasks, specify the language if applicable.
- For math tasks, the problem should be solvable and well-defined.
- For science tasks, be precise about the subfield and concept.
- For creative writing, provide a rich, evocative prompt.
- For conversation tasks, set up a realistic conversational scenario.
- CRITICAL: If the subdomain is "electromagnetism", the task must be about electromagnetism specifically, not general mechanics or optics. If the subdomain is "sorting", the task must involve sorting algorithms, not graph traversal. This strict alignment applies to ALL subdomains."#;

const DONATION_BTC: &str = "bc1qx6zepu6sfkvshgdmc4ewu6pk6rpadvpgffpp7v";
const DONATION_LTC: &str = "ltc1qv2mefzps2vtjcpwfx8xxdrpplrcvltswm68r7x";
const DONATION_XMR: &str = "42Dbm5xg5Nq26fdyzfEU7KBnAJfhi7Cvz5J2ex5CzHXkfKuNEJzYCcmJ1GTbgjFZ5MBx72sdG1G9239Cd6rsZfv4QeDkYJY";

#[derive(Debug, Clone)]
struct DomainDef {
    category: &'static str,
    name: &'static str, 
    subdomains: &'static [&'static str],
}

static DOMAINS: &[DomainDef] = &[
    DomainDef { category: "math", name: "Algebra", subdomains: &["linear_equations", "polynomials", "inequalities", "matrices", "abstract_algebra"] },
    DomainDef { category: "math", name: "Calculus", subdomains: &["derivatives", "integrals", "limits", "series", "multivariable"] },
    DomainDef { category: "math", name: "Probability", subdomains: &["bayesian", "distributions", "combinatorics", "stochastic_processes", "markov_chains"] },
    DomainDef { category: "math", name: "Statistics", subdomains: &["hypothesis_testing", "regression", "anova", "descriptive", "bayesian_stats"] },
    DomainDef { category: "math", name: "Geometry", subdomains: &["euclidean", "analytic", "differential", "topology", "trigonometry"] },
    DomainDef { category: "math", name: "Number Theory", subdomains: &["primes", "modular_arithmetic", "diophantine", "cryptographic", "algebraic_nt"] },
    DomainDef { category: "math", name: "Discrete Math", subdomains: &["graph_theory", "combinatorics", "logic", "set_theory", "recurrence"] },
    DomainDef { category: "math", name: "Linear Algebra", subdomains: &["vector_spaces", "eigenvalues", "transformations", "inner_product", "decompositions"] },
    DomainDef { category: "coding", name: "Web Development", subdomains: &["html_css", "javascript", "react", "nodejs", "rest_apis", "databases", "authentication", "websockets"] },
    DomainDef { category: "coding", name: "C++", subdomains: &["templates", "stl", "memory_management", "concurrency", "meta_programming", "algorithms"] },
    DomainDef { category: "coding", name: "Java", subdomains: &["spring", "concurrency", "jvm", "design_patterns", "streams", "generics"] },
    DomainDef { category: "coding", name: "JavaScript", subdomains: &["async_await", "closures", "prototypes", "modules", "typescript", "dom", "node"] },
    DomainDef { category: "coding", name: "C", subdomains: &["pointers", "memory", "systems_programming", "ffi", "embedded", "algorithms"] },
    DomainDef { category: "coding", name: "Ruby", subdomains: &["metaprogramming", "rails", "blocks_procs", "concurrency", "gems", "dsls"] },
    DomainDef { category: "coding", name: "Lua", subdomains: &["coroutines", "metatable", "game_scripting", "embedded", "neovim", "love2d"] },
    DomainDef { category: "coding", name: "Rust", subdomains: &["ownership", "lifetimes", "traits", "async", "unsafe", "macros", "cargo"] },
    DomainDef { category: "coding", name: "C#", subdomains: &["linq", "async", "unity", "dotnet", "generics", "reflection", "entity_framework"] },
    DomainDef { category: "coding", name: "Python", subdomains: &["data_science", "ml", "web_frameworks", "scripting", "asyncio", "decorators"] },
    DomainDef { category: "coding", name: "Go", subdomains: &["goroutines", "channels", "interfaces", "modules", "networking", "microservices"] },
    DomainDef { category: "coding", name: "SQL", subdomains: &["queries", "optimization", "schema_design", "transactions", "window_functions", "nosql_comparison"] },
    DomainDef { category: "science", name: "Physics", subdomains: &["mechanics", "thermodynamics", "electromagnetism", "quantum", "relativity", "optics", "fluid_dynamics"] },
    DomainDef { category: "science", name: "Chemistry", subdomains: &["organic", "inorganic", "physical", "analytical", "biochemistry", "electrochemistry"] },
    DomainDef { category: "science", name: "Biology", subdomains: &["genetics", "ecology", "cell_biology", "evolution", "microbiology", "neuroscience", "immunology"] },
    DomainDef { category: "science", name: "Earth Science", subdomains: &["geology", "meteorology", "oceanography", "climate", "mineralogy", "volcanology"] },
    DomainDef { category: "science", name: "Astronomy", subdomains: &["stellar", "planetary", "cosmology", "astrophysics", "observational", "astrobiology"] },
    DomainDef { category: "cs", name: "Algorithms", subdomains: &["sorting", "searching", "dynamic_programming", "greedy", "divide_conquer", "graph_algorithms"] },
    DomainDef { category: "cs", name: "Data Structures", subdomains: &["trees", "graphs", "hash_tables", "heaps", "tries", "bloom_filters"] },
    DomainDef { category: "cs", name: "Operating Systems", subdomains: &["processes", "memory_management", "filesystems", "scheduling", "virtualization", "concurrency"] },
    DomainDef { category: "cs", name: "Networking", subdomains: &["tcp_ip", "dns", "http", "routing", "security", "wireless", "load_balancing"] },
    DomainDef { category: "cs", name: "Databases", subdomains: &["relational", "nosql", "indexing", "transactions", "distributed", "query_optimization"] },
    DomainDef { category: "cs", name: "Compilers", subdomains: &["lexing", "parsing", "ast", "code_gen", "optimization", "type_checking"] },
    DomainDef { category: "cs", name: "Distributed Systems", subdomains: &["consensus", "replication", "partitioning", "cap_theorem", "mapreduce", "eventual_consistency"] },
    DomainDef { category: "cs", name: "Machine Learning", subdomains: &["supervised", "unsupervised", "reinforcement", "neural_networks", "transformers", "evaluation"] },
    DomainDef { category: "cs", name: "Cybersecurity", subdomains: &["cryptography", "network_security", "web_security", "reverse_engineering", "forensics", "malware_analysis"] },
    DomainDef { category: "cs", name: "Software Engineering", subdomains: &["testing", "ci_cd", "design_patterns", "agile", "refactoring", "documentation"] },
    DomainDef { category: "creative", name: "Fiction", subdomains: &["short_story", "flash_fiction", "dialogue", "worldbuilding", "character_dev", "plot_twist"] },
    DomainDef { category: "creative", name: "Poetry", subdomains: &["sonnet", "free_verse", "haiku", "narrative", "spoken_word", "lyric"] },
    DomainDef { category: "creative", name: "Screenwriting", subdomains: &["dialogue", "scene", "monologue", "plot_structure", "character_arc", "formatting"] },
    DomainDef { category: "creative", name: "Journalism", subdomains: &["investigative", "feature", "opinion", "reporting", "interview", "editorial"] },
    DomainDef { category: "creative", name: "Songwriting", subdomains: &["lyrics", "melody_description", "concept", "hook", "verse_chorus", "story_song"] },
    DomainDef { category: "creative", name: "Game Narrative", subdomains: &["quest_design", "dialogue_trees", "lore", "cutscene", "branching_narrative", "environmental_storytelling"] },
    DomainDef { category: "creative", name: "Copywriting", subdomains: &["ad_copy", "slogans", "product_descriptions", "email_marketing", "social_media", "brand_voice"] },
    DomainDef { category: "creative", name: "Blogging", subdomains: &["how_to", "listicle", "opinion", "tutorial", "review", "personal_essay"] },
    DomainDef { category: "conversation", name: "Debate", subdomains: &["formal", "casual", "philosophical", "scientific", "political", "ethical"] },
    DomainDef { category: "conversation", name: "Advice", subdomains: &["career", "relationship", "academic", "financial", "health", "technical"] },
    DomainDef { category: "conversation", name: "Interview", subdomains: &["job", "podcast", "journalistic", "research", "behavioral", "technical_interview"] },
    DomainDef { category: "conversation", name: "Teaching", subdomains: &["socratic", "mentoring", "tutoring", "lecture_qa", "feedback", "study_guidance"] },
    DomainDef { category: "conversation", name: "Roleplay", subdomains: &["historical_figure", "professional", "customer_service", "therapy", "negotiation", "collaborative"] },
];

const DEFAULT_DISTRIBUTION: &[(&str, f64)] = &[
    ("math", 0.25),
    ("coding", 0.25),
    ("science", 0.15),
    ("cs", 0.15),
    ("creative", 0.10),
    ("conversation", 0.10),
];

const DEFAULT_DIFFICULTY: &[(u8, f64)] = &[
    (1, 0.05),
    (2, 0.05),
    (3, 0.10),
    (4, 0.15),
    (5, 0.20),
    (6, 0.15),
    (7, 0.10),
    (8, 0.08),
    (9, 0.07),
    (10, 0.05),
];

fn difficulty_label(d: u8) -> &'static str {
    match d {
        1 => "Very Easy (child-level)",
        2 => "Easy (elementary)",
        3 => "Basic (middle school)",
        4 => "Intermediate (high school)",
        5 => "Standard (undergraduate intro)",
        6 => "Skilled (undergraduate advanced)",
        7 => "Proficient (graduate level)",
        8 => "Advanced (professional/researcher)",
        9 => "Expert (top specialist)",
        10 => "Polymath (1-in-a-million genius)",
        _ => "Unknown",
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "taskgen", version, about = "SFT task generator for distillation datasets by empero-org")]
struct Args {
    #[arg(long, default_value = "https://api.openai.com/v1")]
    api_base: String,

    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: Option<String>,

    #[arg(short, long, default_value = "gpt-4o-mini")]
    model: String,

    #[arg(long)]
    system_prompt: Option<String>,

    #[arg(short, long, default_value_t = 250)]
    count: usize,

    #[arg(long)]
    distribution: Option<String>,

    #[arg(long)]
    difficulty: Option<String>,

    #[arg(short, long, default_value_t = 0.9)]
    temperature: f64,

    #[arg(short, long, default_value_t = 5)]
    workers: usize,

    #[arg(short, long, default_value = "output.jsonl")]
    output: PathBuf,

    #[arg(long)]
    append: bool,

    #[arg(long)]
    proxies: Option<PathBuf>,

    #[arg(long)]
    rotating_proxy: bool,

    #[arg(long)]
    input_price: Option<f64>,

    #[arg(long)]
    output_price: Option<f64>,

    #[arg(long)]
    budget: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskEntry {
    prompt: String,
    domain: String,
    subdomain: String,
    difficulty: u8,
    taskgen_model: String,
    temperature: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    usage: Option<Usage>,
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Debug)]
struct RunStats {
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_tasks: usize,
    errors: usize,
}

fn parse_distribution(input: &str) -> Result<HashMap<String, f64>> {
    let mut map = HashMap::new();
    for pair in input.split(',') {
        let pair = pair.trim();
        let (key, val) = pair.split_once('=').context(format!("invalid distribution pair: {}", pair))?;
        let key = key.trim().to_lowercase();
        let val: f64 = val.trim().parse().context(format!("invalid weight: {}", val))?;
        map.insert(key, val);
    }
    let total: f64 = map.values().sum();
    if (total - 1.0).abs() > 0.05 {
        bail!("distribution weights must sum to ~1.0, got {}", total);
    }
    let normalized: HashMap<String, f64> = map.into_iter().map(|(k, v)| (k, v / total)).collect();
    Ok(normalized)
}

fn parse_difficulty(input: &str) -> Result<HashMap<u8, f64>> {
    let mut map = HashMap::new();
    for pair in input.split(',') {
        let pair = pair.trim();
        let (key, val) = pair.split_once('=').context(format!("invalid difficulty pair: {}", pair))?;
        let key: String = key.trim().to_lowercase();
        let d: u8 = if key.starts_with('d') {
            key[1..].parse().context(format!("invalid difficulty level: {}", key))?
        } else {
            key.parse().context(format!("invalid difficulty level: {}", key))?
        };
        if !(1..=10).contains(&d) {
            bail!("difficulty must be 1-10, got {}", d);
        }
        let val: f64 = val.trim().parse().context(format!("invalid weight: {}", val))?;
        map.insert(d, val);
    }
    let total: f64 = map.values().sum();
    if (total - 1.0).abs() > 0.05 {
        bail!("difficulty weights must sum to ~1.0, got {}", total);
    }
    let normalized: HashMap<u8, f64> = map.into_iter().map(|(k, v)| (k, v / total)).collect();
    Ok(normalized)
}

fn parse_proxy_line(line: &str) -> Result<reqwest::Proxy> {
    let line = line.trim();
    let parts: Vec<&str> = line.split(':').collect();
    let proxy_url = match parts.len() {
        2 => format!("http://{}:{}", parts[0], parts[1]),
        4 => format!("http://{}:{}@{}:{}", parts[2], parts[3], parts[0], parts[1]),
        _ => bail!("invalid proxy format '{}', expected host:port or host:port:user:pass", line),
    };
    reqwest::Proxy::all(&proxy_url).context(format!("failed to create proxy from '{}'", line))
}

fn load_proxies(path: &PathBuf) -> Result<Vec<reqwest::Proxy>> {
    let file = File::open(path).context(format!("failed to open proxy file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut proxies = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.context("failed to read proxy file")?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        proxies.push(parse_proxy_line(&line).context(format!("proxy line {}", i + 1))?);
    }
    if proxies.is_empty() {
        bail!("proxy file is empty: {}", path.display());
    }
    Ok(proxies)
}

fn build_clients(proxies: &[reqwest::Proxy]) -> Vec<reqwest::Client> {
    proxies
        .iter()
        .map(|p| {
            reqwest::Client::builder()
                .proxy(p.clone())
                .build()
                .expect("failed to build client with proxy")
        })
        .collect()
}

fn build_domain_pool(dist: &HashMap<String, f64>) -> Vec<(String, String, String, f64)> {
    let mut pool = Vec::new();
    for (cat, &weight) in dist {
        let domains_in_cat: Vec<&DomainDef> = DOMAINS.iter().filter(|d| d.category == cat).collect();
        if domains_in_cat.is_empty() {
            continue;
        }
        let per_domain = weight / domains_in_cat.len() as f64;
        for d in &domains_in_cat {
            for &sub in d.subdomains {
                pool.push((cat.to_string(), d.name.to_string(), sub.to_string(), per_domain / d.subdomains.len() as f64));
            }
        }
    }
    pool
}

fn sample_domain(rng: &mut impl Rng, pool: &[(String, String, String, f64)]) -> (String, String, String) {
    let weights: Vec<f64> = pool.iter().map(|(_, _, _, w)| *w).collect();
    let idx = WeightedIndex::new(&weights).unwrap();
    let i = idx.sample(rng);
    let (cat, name, sub, _) = &pool[i];
    (cat.clone(), name.clone(), sub.clone())
}

fn sample_difficulty(rng: &mut impl Rng, dist: &HashMap<u8, f64>) -> u8 {
    let levels: Vec<u8> = dist.keys().copied().collect();
    let weights: Vec<f64> = levels.iter().map(|l| dist[l]).collect();
    let idx = WeightedIndex::new(&weights).unwrap();
    levels[idx.sample(rng)]
}

async fn generate_task(
    client: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    category: &str,
    domain_display: &str,
    subdomain: &str,
    difficulty: u8,
    temperature: f64,
) -> Result<(String, u64, u64)> {
    let user_msg = format!(
        "Generate a task/prompt for the following:\n\nDomain: {}\nSubdomain: {}\nDifficulty: {}/10 ({})\n\nThe task MUST be directly and specifically about the subdomain \"{}\" within {}. Do NOT generate a generic {} task — the content must focus on {} specifically.\n\nOutput only the task prompt, nothing else.",
        domain_display, subdomain, difficulty, difficulty_label(difficulty),
        subdomain, domain_display, domain_display, subdomain
    );

    let body = ChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage { role: "system".into(), content: system_prompt.into() },
            ChatMessage { role: "user".into(), content: user_msg },
        ],
        temperature,
        max_tokens: Some(2048),
    };

    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .context("API request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("API error {}: {}", status, text);
    }

    let chat_resp: ChatResponse = resp.json().await.context("failed to parse API response")?;
    let choice = chat_resp.choices.into_iter().next().context("no choices in response")?;
    let prompt_text = choice.message.content.trim().to_string();

    let (input_tokens, output_tokens) = match chat_resp.usage {
        Some(u) => (u.prompt_tokens, u.completion_tokens),
        None => (0, 0),
    };

    Ok((prompt_text, input_tokens, output_tokens))
}

fn count_existing_tasks(path: &PathBuf) -> usize {
    if !path.exists() {
        return 0;
    }
    let file = File::open(path).ok();
    match file {
        Some(f) => BufReader::new(f).lines().filter_map(|l| l.ok()).filter(|l| !l.trim().is_empty()).count(),
        None => 0,
    }
}

fn generate_readme(
    args: &Args,
    stats: &RunStats,
    dist: &HashMap<String, f64>,
    diff_dist: &HashMap<u8, f64>,
) -> String {
    let input_cost = args.input_price.map(|p| p * stats.total_input_tokens as f64 / 1_000_000.0);
    let output_cost = args.output_price.map(|p| p * stats.total_output_tokens as f64 / 1_000_000.0);
    let total_cost = match (input_cost, output_cost) {
        (Some(i), Some(o)) => Some(i + o),
        _ => None,
    };

    let mut md = String::new();

    md.push_str("# TaskGen Dataset\n\n");
    md.push_str("> Generated with **taskgen** by [empero-org](https://github.com/empero-org)\n\n");

    md.push_str("## Run Parameters\n\n");
    md.push_str("| Parameter | Value |\n|---|---|\n");
    md.push_str(&format!("| Model | `{}` |\n", args.model));
    md.push_str(&format!("| Temperature | `{}` |\n", args.temperature));
    md.push_str(&format!("| Total Tasks | {} |\n", stats.total_tasks));
    md.push_str(&format!("| Concurrency | {} workers |\n", args.workers));
    md.push_str(&format!("| API Base | `{}` |\n", args.api_base));
    md.push_str(&format!("| Generated | {} |\n", Local::now().format("%Y-%m-%d %H:%M:%S")));
    if let Some(b) = args.budget {
        md.push_str(&format!("| Budget Cap | ${:.4} |\n", b));
    }
    md.push('\n');

    md.push_str("## Domain Distribution\n\n");
    md.push_str("| Domain | Weight |\n|---|---|\n");
    let mut sorted_cats: Vec<_> = dist.iter().collect();
    sorted_cats.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (cat, w) in &sorted_cats {
        md.push_str(&format!("| {} | {:.1}% |\n", cat, **w * 100.0));
    }
    md.push('\n');

    md.push_str("## Difficulty Distribution\n\n");
    md.push_str("| Level | Label | Weight |\n|---|---|---|\n");
    for d in 1..=10u8 {
        if let Some(w) = diff_dist.get(&d) {
            md.push_str(&format!("| {} | {} | {:.1}% |\n", d, difficulty_label(d), w * 100.0));
        }
    }
    md.push('\n');

    md.push_str("## Token Usage & Cost\n\n");
    md.push_str("| Metric | Value |\n|---|---|\n");
    md.push_str(&format!("| Input Tokens | {} |\n", stats.total_input_tokens));
    md.push_str(&format!("| Output Tokens | {} |\n", stats.total_output_tokens));
    md.push_str(&format!("| Total Tokens | {} |\n", stats.total_input_tokens + stats.total_output_tokens));
    md.push_str(&format!("| Errors | {} |\n", stats.errors));
    if let Some(ic) = input_cost {
        md.push_str(&format!("| Input Cost | ${:.6} |\n", ic));
    }
    if let Some(oc) = output_cost {
        md.push_str(&format!("| Output Cost | ${:.6} |\n", oc));
    }
    if let Some(tc) = total_cost {
        md.push_str(&format!("| **Total Cost** | **${:.6}** |\n", tc));
    }
    if args.input_price.is_none() && args.output_price.is_none() {
        md.push_str("| Cost | *Not calculated (use --input-price and --output-price per M tokens)* |\n");
    }
    md.push('\n');

    md.push_str("## Output Format\n\n");
    md.push_str("Each line in the JSONL file contains:\n\n");
    md.push_str("```json\n");
    md.push_str("{\n");
    md.push_str("  \"prompt\": \"...\",\n");
    md.push_str("  \"domain\": \"math::Algebra\",\n");
    md.push_str("  \"subdomain\": \"polynomials\",\n");
    md.push_str("  \"difficulty\": 5,\n");
    md.push_str("  \"taskgen_model\": \"gpt-4o-mini\",\n");
    md.push_str("  \"temperature\": 0.9\n");
    md.push_str("}\n");
    md.push_str("```\n\n");

    md.push_str("## Support / Donate\n\n");
    md.push_str("If this tool helped you, consider supporting the project:\n\n");
    md.push_str(&format!("- **BTC**: `{}`\n", DONATION_BTC));
    md.push_str(&format!("- **LTC**: `{}`\n", DONATION_LTC));
    md.push_str(&format!("- **XMR**: `{}`\n\n", DONATION_XMR));

    md.push_str("---\n\n");
    md.push_str("*Built with [taskgen](https://github.com/empero-org/taskgen) by empero-org*\n");

    md
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let api_key = args.api_key.clone().context("API key required. Use --api-key or set OPENAI_API_KEY env var")?;

    let dist: HashMap<String, f64> = match &args.distribution {
        Some(d) => parse_distribution(d)?,
        None => DEFAULT_DISTRIBUTION.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    };

    let diff_dist: HashMap<u8, f64> = match &args.difficulty {
        Some(d) => parse_difficulty(d)?,
        None => DEFAULT_DIFFICULTY.iter().map(|(k, v)| (*k, *v)).collect(),
    };

    let pool = build_domain_pool(&dist);
    if pool.is_empty() {
        bail!("no domains matched the given distribution. Available categories: math, coding, science, cs, creative, conversation");
    }

    let system_prompt = args.system_prompt.as_deref().unwrap_or(DEFAULT_SYSTEM_PROMPT);

    let existing = if args.append { count_existing_tasks(&args.output) } else { 0 };
    if existing > 0 {
        println!("Appending to existing file with {} tasks", existing);
    }

    let file = if args.append && args.output.exists() {
        OpenOptions::new().append(true).open(&args.output)?
    } else {
        File::create(&args.output)?
    };

    let clients: Arc<Vec<reqwest::Client>> = Arc::new(match &args.proxies {
        Some(proxy_path) => {
            let proxies = load_proxies(proxy_path)?;
            let total = proxies.len();
            if args.rotating_proxy {
                let idx = thread_rng().gen_range(0..total);
                println!("Using rotating proxy (sticky): proxy #{}", idx + 1);
                vec![reqwest::Client::builder()
                    .proxy(proxies.into_iter().nth(idx).unwrap())
                    .build()?]
            } else {
                println!("Loaded {} proxies (round-robin)", total);
                build_clients(&proxies)
            }
        }
        None => vec![reqwest::Client::new()],
    });
    let proxy_counter = Arc::new(AtomicUsize::new(0));

    let file = Arc::new(std::sync::Mutex::new(file));
    let stats = Arc::new(std::sync::Mutex::new(RunStats {
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_tasks: 0,
        errors: 0,
    }));

    let budget = args.budget;
    let input_price = args.input_price;
    let output_price = args.output_price;
    let count = args.count;
    let workers = args.workers;

    println!("Generating {} tasks with {} workers...", count, workers);
    println!("Model: {} | Temperature: {}", args.model, args.temperature);

    let results: Vec<_> = stream::iter(0..count)
        .map(|i| {
            let clients = clients.clone();
            let proxy_counter = proxy_counter.clone();
            let file = file.clone();
            let stats = stats.clone();
            let api_base = args.api_base.clone();
            let api_key = api_key.clone();
            let model = args.model.clone();
            let system_prompt = system_prompt.to_string();
            let pool = pool.clone();
            let diff_dist = diff_dist.clone();
            let temperature = args.temperature;
            let budget = budget;
            let input_price = input_price;
            let output_price = output_price;

            async move {
                let mut rng = thread_rng();
                let (cat, domain_name, subdomain) = sample_domain(&mut rng, &pool);
                let difficulty = sample_difficulty(&mut rng, &diff_dist);

                {
                    let s = stats.lock().unwrap();
                    if let (Some(b), Some(ip), Some(op)) = (budget, input_price, output_price) {
                        let cost = (ip * s.total_input_tokens as f64 / 1_000_000.0)
                            + (op * s.total_output_tokens as f64 / 1_000_000.0);
                        if cost >= b {
                            return;
                        }
                    }
                }

                let domain_display = format!("{}::{}", cat, domain_name);
                let client_idx = proxy_counter.fetch_add(1, Ordering::Relaxed) % clients.len();
                let client = &clients[client_idx];

                match generate_task(
                    client,
                    &api_base,
                    &api_key,
                    &model,
                    &system_prompt,
                    &cat,
                    &domain_display,
                    &subdomain,
                    difficulty,
                    temperature,
                )
                .await
                {
                    Ok((prompt, in_tok, out_tok)) => {
                        if prompt.trim().is_empty() {
                            eprintln!("[WARN] task {}: empty prompt, skipping", i + 1);
                            let mut s = stats.lock().unwrap();
                            s.errors += 1;
                            return;
                        }
                        let entry = TaskEntry {
                            prompt,
                            domain: format!("{}::{}", cat, domain_name),
                            subdomain,
                            difficulty,
                            taskgen_model: model.clone(),
                            temperature,
                        };
                        let line = serde_json::to_string(&entry).unwrap() + "\n";
                        {
                            let mut f = file.lock().unwrap();
                            let _ = f.write_all(line.as_bytes());
                            let _ = f.flush();
                        }
                        let should_print = {
                            let mut s = stats.lock().unwrap();
                            s.total_input_tokens += in_tok;
                            s.total_output_tokens += out_tok;
                            s.total_tasks += 1;
                            (i + 1) % 10 == 0 || i == 0
                        };
                        if should_print {
                            let s = stats.lock().unwrap();
                            println!("[{}/{}] generated ({} errors)", s.total_tasks, count, s.errors);
                        }
                    }
                    Err(e) => {
                        eprintln!("[ERROR] task {}: {}", i + 1, e);
                        let mut s = stats.lock().unwrap();
                        s.errors += 1;
                    }
                }
            }
        })
        .buffer_unordered(workers)
        .collect()
        .await;

    let stats = stats.lock().unwrap();
    println!("\nDone! Generated {} tasks ({} errors)", stats.total_tasks, stats.errors);
    println!("Tokens: {} in / {} out", stats.total_input_tokens, stats.total_output_tokens);

    // Deduplicate output file
    let mut seen: HashSet<String> = HashSet::new();
    let mut deduped_lines: Vec<String> = Vec::new();
    let mut duplicates: usize = 0;

    if args.output.exists() {
        let reader = BufReader::new(File::open(&args.output)?);
        for line in reader.lines().flatten() {
            if let Ok(entry) = serde_json::from_str::<TaskEntry>(&line) {
                let normalized = entry.prompt.to_lowercase();
                let normalized: String = normalized.split_whitespace().collect();
                if seen.insert(normalized) {
                    deduped_lines.push(line);
                } else {
                    duplicates += 1;
                }
            } else {
                deduped_lines.push(line);
            }
        }
    }

    if duplicates > 0 {
        let mut f = File::create(&args.output)?;
        for line in &deduped_lines {
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
        }
        println!("Deduplicated: removed {} duplicate prompts ({} remaining)", duplicates, deduped_lines.len());
    }

    let readme = generate_readme(&args, &stats, &dist, &diff_dist);
    let readme_path = args.output.parent().unwrap_or(std::path::Path::new(".")).join("README.md");
    let mut rf = File::create(&readme_path).context("failed to create README.md")?;
    rf.write_all(readme.as_bytes())?;
    println!("README.md written to {}", readme_path.display());

    Ok(())
}
