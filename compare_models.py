from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Sample data for testing
texts = [
    "album music cd songs like great song love just good",
    "recording quality sound bad recording terrible poor",
    "band rock guitar heavy metal awesome amazing",
    "voice singer vocal beautiful amazing wonderful",
    "jazz blues soul music great love song",
    "pop dance electronic music cool awesome great",
    "classical piano symphony orchestra beautiful",
    "hip hop rap music cool fresh new",
    "country folk guitar singer song love",
    "reggae ska music cool chill vibe"
]

# Tokenize texts
tokenized_texts = [text.split() for text in texts]

# Create dictionary
id2word = Dictionary(tokenized_texts)

print("=== Topic Modeling Comparison ===\n")

# Simulate BERTopic results (10 topics)
bertopic_topics = [
    [("album", 0.8), ("music", 0.7), ("cd", 0.6), ("songs", 0.5), ("like", 0.4)],
    [("recording", 0.9), ("quality", 0.8), ("sound", 0.7), ("bad", 0.6), ("terrible", 0.5)],
    [("band", 0.8), ("rock", 0.7), ("guitar", 0.6), ("heavy", 0.5), ("metal", 0.4)],
    [("voice", 0.9), ("singer", 0.8), ("vocal", 0.7), ("beautiful", 0.6), ("amazing", 0.5)],
    [("jazz", 0.8), ("blues", 0.7), ("soul", 0.6), ("music", 0.5), ("great", 0.4)],
    [("pop", 0.9), ("dance", 0.8), ("electronic", 0.7), ("music", 0.6), ("cool", 0.5)],
    [("classical", 0.8), ("piano", 0.7), ("symphony", 0.6), ("orchestra", 0.5), ("beautiful", 0.4)],
    [("hip", 0.7), ("hop", 0.6), ("rap", 0.5), ("music", 0.4), ("cool", 0.3)],
    [("country", 0.8), ("folk", 0.7), ("guitar", 0.6), ("singer", 0.5), ("song", 0.4)],
    [("reggae", 0.9), ("ska", 0.8), ("music", 0.7), ("cool", 0.6), ("chill", 0.5)]
]

# Simulate FuzzyTM results (10 topics)
fuzzytm_topics = [
    [("album", 0.85), ("music", 0.8), ("cd", 0.75), ("songs", 0.7), ("like", 0.65)],
    [("recording", 0.9), ("quality", 0.85), ("sound", 0.8), ("bad", 0.75), ("terrible", 0.7)],
    [("band", 0.85), ("rock", 0.8), ("guitar", 0.75), ("heavy", 0.7), ("metal", 0.65)],
    [("voice", 0.9), ("singer", 0.85), ("vocal", 0.8), ("beautiful", 0.75), ("amazing", 0.7)],
    [("jazz", 0.85), ("blues", 0.8), ("soul", 0.75), ("music", 0.7), ("great", 0.65)],
    [("pop", 0.9), ("dance", 0.85), ("electronic", 0.8), ("music", 0.75), ("cool", 0.7)],
    [("classical", 0.85), ("piano", 0.8), ("symphony", 0.75), ("orchestra", 0.7), ("beautiful", 0.65)],
    [("hip", 0.8), ("hop", 0.75), ("rap", 0.7), ("music", 0.65), ("cool", 0.6)],
    [("country", 0.85), ("folk", 0.8), ("guitar", 0.75), ("singer", 0.7), ("song", 0.65)],
    [("reggae", 0.9), ("ska", 0.85), ("music", 0.8), ("cool", 0.75), ("chill", 0.7)]
]

# Function to calculate coherence
def calculate_coherence(topics, id2word):
    topic_words = [[word for word, _ in topic] for topic in topics]
    coherence_model = CoherenceModel(topics=topic_words, texts=tokenized_texts, dictionary=id2word, coherence='c_v')
    return coherence_model.get_coherence()

# Function to calculate topic diversity
def calculate_topic_diversity(topics):
    unique_words = set()
    total_words = 0
    for topic in topics:
        for word, _ in topic:
            unique_words.add(word)
            total_words += 1
    return len(unique_words) / total_words if total_words > 0 else 0

# Calculate metrics for both models
print("BERTopic Metrics:")
c_v_bertopic = calculate_coherence(bertopic_topics, id2word)
diversity_bertopic = calculate_topic_diversity(bertopic_topics)
print(f"  C_v Coherence: {c_v_bertopic:.4f}")
print(f"  Topic Diversity: {diversity_bertopic:.4f}")
print()

print("FuzzyTM Metrics:")
c_v_fuzzytm = calculate_coherence(fuzzytm_topics, id2word)
diversity_fuzzytm = calculate_topic_diversity(fuzzytm_topics)
print(f"  C_v Coherence: {c_v_fuzzytm:.4f}")
print(f"  Topic Diversity: {diversity_fuzzytm:.4f}")
print()

# Compare results
print("Comparison Summary:")
print(f"C_v Coherence - BERTopic: {c_v_bertopic:.4f}, FuzzyTM: {c_v_fuzzytm:.4f}")
print(f"Topic Diversity - BERTopic: {diversity_bertopic:.4f}, FuzzyTM: {diversity_fuzzytm:.4f}")

if c_v_fuzzytm > c_v_bertopic:
    print("\nFuzzyTM has better topic coherence")
elif c_v_bertopic > c_v_fuzzytm:
    print("\nBERTopic has better topic coherence")
else:
    print("\nBoth models have similar topic coherence")

if diversity_fuzzytm > diversity_bertopic:
    print("FuzzyTM has better topic diversity")
elif diversity_bertopic > diversity_fuzzytm:
    print("BERTopic has better topic diversity")
else:
    print("Both models have similar topic diversity")
