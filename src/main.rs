use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use chrono::Local;
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
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

const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("nl", "Dutch"),
    ("zh", "Chinese"),
    ("ar", "Arabic"),
    ("ru", "Russian"),
];

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
    keyfile: Option<PathBuf>,

    #[arg(long)]
    dedup: bool,

    #[arg(long, default_value_t = 0.6)]
    dedup_threshold: f64,

    #[arg(long)]
    free_models: bool,

    /// Rescan interval in minutes for free model availability (default: 10)
    #[arg(long, default_value_t = 10)]
    free_rescan: u64,

    #[arg(long)]
    input_price: Option<f64>,

    #[arg(long)]
    output_price: Option<f64>,

    #[arg(long)]
    budget: Option<f64>,

    /// Generate tasks in multiple languages (en, de, fr, es, nl, zh, ar, ru)
    #[arg(long)]
    multilingual: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskEntry {
    prompt: String,
    domain: String,
    subdomain: String,
    difficulty: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
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

const MAX_MODEL_FAILURES: usize = 3;

struct ModelFailures {
    counts: std::sync::Mutex<HashMap<String, usize>>,
    rescan_notify: tokio::sync::Notify,
}

impl ModelFailures {
    fn new() -> Self {
        Self {
            counts: std::sync::Mutex::new(HashMap::new()),
            rescan_notify: tokio::sync::Notify::new(),
        }
    }

    /// Record a failure. Returns true if the model just crossed the threshold.
    fn record(&self, model: &str) -> bool {
        let mut counts = self.counts.lock().unwrap();
        let count = counts.entry(model.to_string()).or_insert(0);
        *count += 1;
        *count == MAX_MODEL_FAILURES
    }

    /// Remove a model from tracking (called after rescan replaces the list).
    fn reset(&self) {
        let mut counts = self.counts.lock().unwrap();
        counts.clear();
    }
}

struct AtomicStats {
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    tasks: AtomicUsize,
    errors: AtomicUsize,
}

impl AtomicStats {
    fn new() -> Self {
        Self {
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
            tasks: AtomicUsize::new(0),
            errors: AtomicUsize::new(0),
        }
    }
}

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

fn word_trigrams(text: &str) -> HashSet<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return words.iter().map(|w| w.to_string()).collect();
    }
    words.windows(3).map(|w| w.join(" ")).collect()
}

fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 { return 0.0; }
    intersection as f64 / union as f64
}

fn load_api_keys(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path).context(format!("failed to open keyfile: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut keys = Vec::new();
    for line in reader.lines() {
        let line = line.context("failed to read keyfile")?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        keys.push(line);
    }
    if keys.is_empty() {
        bail!("keyfile is empty: {}", path.display());
    }
    Ok(keys)
}

const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
const MIN_FREE_MODEL_CTX: u64 = 16000;

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    name: String,
    architecture: ModelArchitecture,
    pricing: ModelPricing,
    top_provider: ModelProvider,
}

#[derive(Debug, Deserialize)]
struct ModelArchitecture {
    input_modalities: Vec<String>,
    output_modalities: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelPricing {
    prompt: String,
    completion: String,
}

#[derive(Debug, Deserialize)]
struct ModelProvider {
    context_length: Option<u64>,
    max_completion_tokens: Option<u64>,
}

async fn fetch_free_models(client: &reqwest::Client, api_key: &str) -> Result<Vec<String>> {
    let url = format!("{}/models", OPENROUTER_API_BASE);
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
        .context("failed to fetch OpenRouter models")?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        bail!("OpenRouter models API error: {}", text);
    }

    let models: ModelsResponse = resp.json().await.context("failed to parse models response")?;

    let mut free: Vec<(String, String, u64)> = models
        .data
        .into_iter()
        .filter(|m| {
            m.pricing.prompt == "0"
                && m.pricing.completion == "0"
                && m.architecture.input_modalities.contains(&"text".to_string())
                && m.architecture.output_modalities.contains(&"text".to_string())
                && m.id != "openrouter/free"
                && m.top_provider.context_length.unwrap_or(0) >= MIN_FREE_MODEL_CTX
        })
        .map(|m| {
            let ctx = m.top_provider.context_length.unwrap_or(0);
            (m.id, m.name, ctx)
        })
        .collect();

    // sort by context length descending so best models are first
    free.sort_by(|a, b| b.2.cmp(&a.2));

    if free.is_empty() {
        bail!("no free models with >= {}k context available on OpenRouter right now", MIN_FREE_MODEL_CTX / 1000);
    }

    println!("Found {} candidate free models, running health checks...", free.len());

    // ping each model with a tiny request to verify it's actually online
    let mut verified: Vec<String> = Vec::new();
    for (id, name, ctx) in &free {
        print!("  testing {} ({}, {}k ctx)... ", id, name, ctx / 1000);
        match test_model(client, api_key, id).await {
            Ok(()) => {
                println!("ok");
                verified.push(id.clone());
            }
            Err(e) => {
                println!("offline ({})", e);
            }
        }
    }

    if verified.is_empty() {
        bail!("all free models are offline on OpenRouter right now");
    }

    println!("Using {} verified free models", verified.len());
    Ok(verified)
}

async fn test_model(client: &reqwest::Client, api_key: &str, model: &str) -> Result<()> {
    let body = ChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Say hi.".into(),
        }],
        temperature: 0.0,
        max_tokens: Some(5),
    };

    let url = format!("{}/chat/completions", OPENROUTER_API_BASE);
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .context("request failed")?;

    let status = resp.status();

    // 429 means the model exists and is live, just rate limited — count as available
    if status.as_u16() == 429 {
        return Ok(());
    }

    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        bail!("{}: {}", status, &text[..text.len().min(100)]);
    }

    let chat_resp: ChatResponse = resp.json().await.context("bad response")?;
    if chat_resp.choices.is_empty() {
        bail!("no choices returned");
    }

    Ok(())
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

enum ApiError {
    RateLimit(Option<u64>),
    Billing(String),
    Timeout,
    Other(anyhow::Error),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::RateLimit(s) => write!(f, "rate limited (retry after {:?}s)", s),
            ApiError::Billing(msg) => write!(f, "billing error: {}", msg),
            ApiError::Timeout => write!(f, "request timed out"),
            ApiError::Other(e) => write!(f, "{}", e),
        }
    }
}

fn is_billing_error(status: reqwest::StatusCode, body: &str) -> bool {
    if status.as_u16() == 402 {
        return true;
    }
    let lower = body.to_lowercase();
    lower.contains("insufficient_quota")
        || lower.contains("billing")
        || lower.contains("payment required")
        || lower.contains("exceeded your current quota")
        || lower.contains("account is not active")
        || lower.contains("insufficient_funds")
        || lower.contains("budget")
}

async fn api_request(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &ChatRequest,
) -> std::result::Result<(String, u64, u64), ApiError> {
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .await;

    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            if e.is_timeout() {
                return Err(ApiError::Timeout);
            }
            return Err(ApiError::Other(e.into()));
        }
    };

    let status = resp.status();

    if status.as_u16() == 429 {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        return Err(ApiError::RateLimit(retry_after));
    }

    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        if is_billing_error(status, &text) {
            return Err(ApiError::Billing(text));
        }
        return Err(ApiError::Other(anyhow::anyhow!("API error {}: {}", status, text)));
    }

    let chat_resp: ChatResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::Other(anyhow::anyhow!("failed to parse API response: {}", e)))?;
    let choice = chat_resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Other(anyhow::anyhow!("no choices in response")))?;
    let prompt_text = choice.message.content.trim().to_string();

    let (input_tokens, output_tokens) = match chat_resp.usage {
        Some(u) => (u.prompt_tokens, u.completion_tokens),
        None => (0, 0),
    };

    Ok((prompt_text, input_tokens, output_tokens))
}

const MAX_RETRIES: u32 = 5;

async fn generate_task(
    client: &reqwest::Client,
    api_base: &str,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    _category: &str,
    domain_display: &str,
    subdomain: &str,
    difficulty: u8,
    temperature: f64,
    language: Option<&str>,
    cancel: &AtomicBool,
    consecutive_timeouts: &AtomicUsize,
    pb: &ProgressBar,
) -> std::result::Result<(String, u64, u64), ApiError> {
    let lang_instruction = match language {
        Some(code) if code != "en" => {
            let lang_name = LANGUAGES.iter().find(|(c, _)| *c == code).map(|(_, n)| *n).unwrap_or("English");
            format!("\n\nIMPORTANT: Write the entire task/prompt in {}. Do NOT use English.", lang_name)
        }
        _ => String::new(),
    };

    let user_msg = format!(
        "Generate a task/prompt for the following:\n\nDomain: {}\nSubdomain: {}\nDifficulty: {}/10 ({})\n\nThe task MUST be directly and specifically about the subdomain \"{}\" within {}. Do NOT generate a generic {} task — the content must focus on {} specifically.\n\nOutput only the task prompt, nothing else.{}",
        domain_display, subdomain, difficulty, difficulty_label(difficulty),
        subdomain, domain_display, domain_display, subdomain,
        lang_instruction
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

    let mut retries = 0u32;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(ApiError::Other(anyhow::anyhow!("cancelled")));
        }

        match api_request(client, &url, api_key, &body).await {
            Ok(result) => {
                consecutive_timeouts.store(0, Ordering::Relaxed);
                return Ok(result);
            }
            Err(ApiError::RateLimit(retry_after)) => {
                retries += 1;
                if retries > MAX_RETRIES {
                    return Err(ApiError::RateLimit(retry_after));
                }
                let wait = retry_after.unwrap_or_else(|| 2u64.pow(retries).min(60));
                pb.suspend(|| {
                    eprintln!("[RATE] 429 hit, waiting {}s (retry {}/{})", wait, retries, MAX_RETRIES);
                });
                tokio::time::sleep(tokio::time::Duration::from_secs(wait)).await;
            }
            Err(ApiError::Timeout) => {
                let count = consecutive_timeouts.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= 5 {
                    pb.suspend(|| {
                        eprintln!("[FATAL] {} consecutive timeouts, shutting down gracefully...", count);
                    });
                    cancel.store(true, Ordering::Relaxed);
                    return Err(ApiError::Timeout);
                }
                retries += 1;
                if retries > MAX_RETRIES {
                    return Err(ApiError::Timeout);
                }
                let wait = 2u64.pow(retries).min(30);
                pb.suspend(|| {
                    eprintln!("[TIMEOUT] request timed out, waiting {}s (retry {}/{}, {} consecutive)", wait, retries, MAX_RETRIES, count);
                });
                tokio::time::sleep(tokio::time::Duration::from_secs(wait)).await;
            }
            Err(ApiError::Billing(msg)) => {
                pb.suspend(|| {
                    eprintln!("[FATAL] billing error, shutting down gracefully: {}", msg);
                });
                cancel.store(true, Ordering::Relaxed);
                return Err(ApiError::Billing(msg));
            }
            Err(e) => return Err(e),
        }
    }
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
    lang_counts: Option<&HashMap<String, usize>>,
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
    if args.multilingual {
        md.push_str("| Multilingual | Yes (en, de, fr, es, nl, zh, ar, ru) |\n");
    }
    md.push('\n');

    if let Some(counts) = lang_counts {
        md.push_str("## Language Distribution\n\n");
        md.push_str("| Language | Code | Tasks |\n|---|---|---|\n");
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (code, count) in &sorted {
            let name = LANGUAGES.iter().find(|(c, _)| *c == code.as_str()).map(|(_, n)| *n).unwrap_or("Unknown");
            md.push_str(&format!("| {} | `{}` | {} |\n", name, code, count));
        }
        md.push('\n');
    }

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
    if args.multilingual {
        md.push_str("  \"language\": \"en\",\n");
    }
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

    let api_keys: Arc<Vec<String>> = Arc::new(match &args.keyfile {
        Some(path) => {
            let keys = load_api_keys(path)?;
            println!("Loaded {} API keys (round-robin)", keys.len());
            keys
        }
        None => {
            let key = args.api_key.clone().context("API key required. Use --api-key, set OPENAI_API_KEY, or use --keyfile")?;
            vec![key]
        }
    });
    let key_counter = Arc::new(AtomicUsize::new(0));

    // discover free models from OpenRouter if requested
    let api_base = if args.free_models {
        OPENROUTER_API_BASE.to_string()
    } else {
        args.api_base.clone()
    };

    let model_failures = Arc::new(ModelFailures::new());

    let free_model_list: Option<Arc<tokio::sync::RwLock<Vec<String>>>> = if args.free_models {
        let discovery_client = reqwest::Client::new();
        let models = fetch_free_models(&discovery_client, &api_keys[0]).await?;
        Some(Arc::new(tokio::sync::RwLock::new(models)))
    } else {
        None
    };
    let model_counter = Arc::new(AtomicUsize::new(0));

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
    let stats = Arc::new(AtomicStats::new());
    let cancel = Arc::new(AtomicBool::new(false));
    let consecutive_timeouts = Arc::new(AtomicUsize::new(0));

    let budget = args.budget;
    let input_price = args.input_price;
    let output_price = args.output_price;
    let count = args.count;
    let workers = args.workers;

    // pre-sample all domain/difficulty/language tuples to avoid RNG contention in workers
    let mut rng = thread_rng();
    let multilingual = args.multilingual;
    let presampled: Vec<(String, String, String, u8, Option<String>)> = (0..count)
        .map(|_| {
            let (cat, name, sub) = sample_domain(&mut rng, &pool);
            let diff = sample_difficulty(&mut rng, &diff_dist);
            let lang = if multilingual {
                let idx = rng.gen_range(0..LANGUAGES.len());
                Some(LANGUAGES[idx].0.to_string())
            } else {
                None
            };
            (cat, name, sub, diff, lang)
        })
        .collect();

    let presampled = Arc::new(presampled);

    let pb = ProgressBar::new(count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) | {msg}")
            .unwrap()
            .progress_chars("##-"),
    );
    pb.set_message("starting...");

    // spawn background rescan task for free models
    let rescan_handle = if let Some(ref model_list) = free_model_list {
        let model_list = model_list.clone();
        let cancel = cancel.clone();
        let model_failures = model_failures.clone();
        let api_key = api_keys[0].clone();
        let rescan_mins = args.free_rescan;
        let pb = pb.clone();
        Some(tokio::spawn(async move {
            let client = reqwest::Client::new();
            loop {
                // wait for either the timer or an immediate rescan trigger
                tokio::select! {
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(rescan_mins * 60)) => {},
                    _ = model_failures.rescan_notify.notified() => {},
                }
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                pb.suspend(|| println!("[RESCAN] refreshing free model list..."));
                match fetch_free_models(&client, &api_key).await {
                    Ok(new_models) => {
                        let count = new_models.len();
                        model_failures.reset();
                        let mut list = model_list.write().await;
                        *list = new_models;
                        pb.suspend(|| println!("[RESCAN] updated: {} models available", count));
                    }
                    Err(e) => {
                        pb.suspend(|| eprintln!("[RESCAN] failed to refresh: {}, keeping current list", e));
                    }
                }
            }
        }))
    } else {
        None
    };

    stream::iter(0..count)
        .for_each_concurrent(workers, |i| {
            let clients = clients.clone();
            let proxy_counter = proxy_counter.clone();
            let file = file.clone();
            let stats = stats.clone();
            let cancel = cancel.clone();
            let consecutive_timeouts = consecutive_timeouts.clone();
            let api_base = api_base.clone();
            let api_keys = api_keys.clone();
            let key_counter = key_counter.clone();
            let model = args.model.clone();
            let free_model_list = free_model_list.clone();
            let model_counter = model_counter.clone();
            let model_failures = model_failures.clone();
            let system_prompt = system_prompt.to_string();
            let presampled = presampled.clone();
            let temperature = args.temperature;
            let pb = pb.clone();

            async move {
                if cancel.load(Ordering::Relaxed) {
                    pb.inc(1);
                    return;
                }

                let (ref cat, ref domain_name, ref subdomain, difficulty, ref lang) = presampled[i];

                if let (Some(b), Some(ip), Some(op)) = (budget, input_price, output_price) {
                    let in_tok = stats.input_tokens.load(Ordering::Relaxed) as f64;
                    let out_tok = stats.output_tokens.load(Ordering::Relaxed) as f64;
                    let cost = (ip * in_tok / 1_000_000.0) + (op * out_tok / 1_000_000.0);
                    if cost >= b {
                        pb.inc(1);
                        return;
                    }
                }

                let use_model = match &free_model_list {
                    Some(models) => {
                        let list = models.read().await;
                        let idx = model_counter.fetch_add(1, Ordering::Relaxed) % list.len();
                        list[idx].clone()
                    }
                    None => model.clone(),
                };

                let domain_display = format!("{}::{}", cat, domain_name);
                let client_idx = proxy_counter.fetch_add(1, Ordering::Relaxed) % clients.len();
                let client = &clients[client_idx];
                let key_idx = key_counter.fetch_add(1, Ordering::Relaxed) % api_keys.len();
                let api_key = &api_keys[key_idx];

                match generate_task(
                    client,
                    &api_base,
                    api_key,
                    &use_model,
                    &system_prompt,
                    cat,
                    &domain_display,
                    subdomain,
                    difficulty,
                    temperature,
                    lang.as_deref(),
                    &cancel,
                    &consecutive_timeouts,
                    &pb,
                )
                .await
                {
                    Ok((prompt, in_tok, out_tok)) => {
                        if prompt.trim().is_empty() {
                            stats.errors.fetch_add(1, Ordering::Relaxed);
                            pb.inc(1);
                            return;
                        }
                        let entry = TaskEntry {
                            prompt,
                            domain: format!("{}::{}", cat, domain_name),
                            subdomain: subdomain.clone(),
                            difficulty,
                            language: lang.clone(),
                            taskgen_model: use_model,
                            temperature,
                        };
                        let line = serde_json::to_string(&entry).unwrap() + "\n";
                        {
                            let mut f = file.lock().unwrap();
                            let _ = f.write_all(line.as_bytes());
                            let _ = f.flush();
                        }
                        stats.input_tokens.fetch_add(in_tok, Ordering::Relaxed);
                        stats.output_tokens.fetch_add(out_tok, Ordering::Relaxed);
                        let done = stats.tasks.fetch_add(1, Ordering::Relaxed) + 1;
                        let errs = stats.errors.load(Ordering::Relaxed);
                        let total_tok = stats.input_tokens.load(Ordering::Relaxed)
                            + stats.output_tokens.load(Ordering::Relaxed);
                        pb.set_message(format!(
                            "{} ok | {} err | {}k tok",
                            done, errs, total_tok / 1000
                        ));
                    }
                    Err(e) => {
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                        if !cancel.load(Ordering::Relaxed) {
                            pb.suspend(|| eprintln!("[ERROR] task {}: {}", i + 1, e));
                        }
                        // track per-model failures for free model rotation
                        if free_model_list.is_some() {
                            let tripped = model_failures.record(&use_model);
                            if tripped {
                                pb.suspend(|| {
                                    eprintln!(
                                        "[RESCAN] {} failed {} times, marking offline and triggering rescan",
                                        use_model, MAX_MODEL_FAILURES
                                    );
                                });
                                model_failures.rescan_notify.notify_one();
                            }
                        }
                    }
                }
                pb.inc(1);
            }
        })
        .await;

    // stop the rescan task
    if let Some(handle) = rescan_handle {
        handle.abort();
    }

    let was_cancelled = cancel.load(Ordering::Relaxed);
    if was_cancelled {
        pb.finish_with_message("stopped early — saving progress");
    } else {
        pb.finish_with_message("done");
    }

    let total_tasks = stats.tasks.load(Ordering::Relaxed);
    let total_errors = stats.errors.load(Ordering::Relaxed);
    let total_in = stats.input_tokens.load(Ordering::Relaxed);
    let total_out = stats.output_tokens.load(Ordering::Relaxed);

    if was_cancelled {
        println!("\nGraceful shutdown — saved {} tasks before exit", total_tasks);
    }
    println!("Generated {} tasks ({} errors)", total_tasks, total_errors);
    println!("Tokens: {} in / {} out", total_in, total_out);

    let stats = RunStats {
        total_input_tokens: total_in,
        total_output_tokens: total_out,
        total_tasks,
        errors: total_errors,
    };

    if args.dedup && args.output.exists() {
        println!("\nRunning deduplication (threshold: {:.2})...", args.dedup_threshold);

        let reader = BufReader::new(File::open(&args.output)?);
        let mut lines: Vec<String> = Vec::new();
        let mut entries: Vec<Option<TaskEntry>> = Vec::new();

        for line in reader.lines().flatten() {
            let entry = serde_json::from_str::<TaskEntry>(&line).ok();
            entries.push(entry);
            lines.push(line);
        }

        // pass 1: exact duplicates
        let mut seen: HashSet<String> = HashSet::new();
        let mut keep = vec![true; lines.len()];
        let mut exact_dupes = 0usize;

        for (i, entry) in entries.iter().enumerate() {
            if let Some(e) = entry {
                let normalized: String = e.prompt.to_lowercase().split_whitespace().collect();
                if !seen.insert(normalized) {
                    keep[i] = false;
                    exact_dupes += 1;
                }
            }
        }

        if exact_dupes > 0 {
            println!("Removed {} exact duplicates", exact_dupes);
        }

        // pass 2: semantic duplicates via word-trigram jaccard
        let kept_indices: Vec<usize> = (0..lines.len()).filter(|&i| keep[i]).collect();
        let trigrams: Vec<Option<HashSet<String>>> = kept_indices
            .iter()
            .map(|&i| {
                entries[i]
                    .as_ref()
                    .map(|e| word_trigrams(&e.prompt.to_lowercase()))
            })
            .collect();

        let mut semantic_dupes = 0usize;
        for j in 1..kept_indices.len() {
            if !keep[kept_indices[j]] {
                continue;
            }
            let trig_b = match &trigrams[j] {
                Some(t) => t,
                None => continue,
            };
            for k in 0..j {
                if !keep[kept_indices[k]] {
                    continue;
                }
                let trig_a = match &trigrams[k] {
                    Some(t) => t,
                    None => continue,
                };
                if jaccard_similarity(trig_a, trig_b) >= args.dedup_threshold {
                    keep[kept_indices[j]] = false;
                    semantic_dupes += 1;
                    break;
                }
            }
        }

        if semantic_dupes > 0 {
            println!("Removed {} semantic duplicates (similarity >= {:.2})", semantic_dupes, args.dedup_threshold);
        }

        let total_removed = exact_dupes + semantic_dupes;
        if total_removed > 0 {
            let mut f = File::create(&args.output)?;
            for (i, line) in lines.iter().enumerate() {
                if keep[i] {
                    f.write_all(line.as_bytes())?;
                    f.write_all(b"\n")?;
                }
            }
            let remaining = lines.len() - total_removed;
            println!("Deduplication complete: {} removed, {} remaining", total_removed, remaining);
        } else {
            println!("No duplicates found");
        }
    }

    // split output into per-language files when --multilingual is set
    let lang_counts: Option<HashMap<String, usize>> = if multilingual && args.output.exists() {
        println!("\nSplitting output by language...");
        let reader = BufReader::new(File::open(&args.output)?);
        let mut lang_buckets: HashMap<String, Vec<String>> = HashMap::new();

        for line in reader.lines().flatten() {
            let lang_code = serde_json::from_str::<serde_json::Value>(&line)
                .ok()
                .and_then(|v| v.get("language").and_then(|l| l.as_str().map(|s| s.to_string())))
                .unwrap_or_else(|| "en".to_string());
            lang_buckets.entry(lang_code).or_default().push(line);
        }

        let out_dir = args.output.parent().unwrap_or(std::path::Path::new("."));
        let stem = args.output.file_stem().unwrap_or_default().to_string_lossy();
        let ext = args.output.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();

        let counts: HashMap<String, usize> = lang_buckets.iter().map(|(k, v)| (k.clone(), v.len())).collect();

        for (lang, lines) in &lang_buckets {
            let lang_path = out_dir.join(format!("{}_{}{}", stem, lang, ext));
            let mut f = File::create(&lang_path)?;
            for line in lines {
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
            }
            println!("  {} — {} tasks -> {}", lang, lines.len(), lang_path.display());
        }

        Some(counts)
    } else {
        None
    };

    let readme = generate_readme(&args, &stats, &dist, &diff_dist, lang_counts.as_ref());
    let readme_path = args.output.parent().unwrap_or(std::path::Path::new(".")).join("README.md");
    let mut rf = File::create(&readme_path).context("failed to create README.md")?;
    rf.write_all(readme.as_bytes())?;
    println!("README.md written to {}", readme_path.display());

    Ok(())
}
