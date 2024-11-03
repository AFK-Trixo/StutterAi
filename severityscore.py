import re

# Expanded transcripts with 2-3 sentences and various stuttering patterns
transcripts = [
    "My-my name is Faaaaris Ziyad. I-I am from-from Texas. This issss a really nice plaaaace.",
    "I-I-I really, um, don't know what to say. Can-can you help me out, uh, with this? It's-it's very confusing.",
    "Can-can you, uh, tell me where the store is? I think it's close by, but-but I'm not sure. Uh, maybe-maybe you can point it out?",
    "This issss really coooool. I'm sooo excited to-to be here. Everyone seems so nice.",
    "My-my-my name is Ali, and I-I work as a teacher. Every day I get to-to meet new people, which is nice.",
    "Hello, how are you today? I'm doing well, thanks for asking! Just here enjoying the day.",
    "Thisss is amazing but, um, I don't kn-know if-if I can do it. It's-it's really challenging for me. Maybe-maybe I need more practice.",
    "Let me, um, let-let me check the tiiiiime quickly. I think we're running late-late. Do you know what time it is?",
    "Uh, my-my-my phone isssss ringing. I should probably answer it-it quickly. I think it's my-my friend calling.",
    "She said that she wou-would be here soooon. I guess we-we can wait a bit longer. It's not a problem.",
    "Do you, um, do you think it's possssiblle to-to-to finish today? I mean, there's a lot left to do. Maybe we-we should come back tomorrow?",
    "I believe we can finish the task today without any problem. It should be straightforward if we all work together.",
    "The weather is beautiful today. I went out for a walk, and it was really nice. I enjoyed the fresh air.",
    "Thank you for the opportunity. I'm excited to start this new project. Looking forward to working with everyone.",
]

# Define stuttering types with regex patterns
stuttering_types = {
    "repetition": r"\b(\w+)-\1\b",
    "block": r"\b\w+ ([a-zA-Z])\b",  # pattern for block (e.g., "k-now")
    "prolongation": r"(\w)\1{2,}",  # repeated characters
    "interjection": r"\b(um|uh|like|you know)\b",
    "natural_pause": r"\.\.\.|--"
}

# Increased severity weights for each type to better align with short transcripts
severity_weights = {
    "repetition": 5,
    "block": 7,
    "prolongation": 6,
    "interjection": 4,
    "natural_pause": 2,
}

# SSI-4 severity levels based on total score ranges
severity_levels = [
    (10, 12, "Very mild"),
    (13, 17, "Mild"),
    (18, 20, "Mild"),
    (21, 24, "Moderate"),
    (25, 27, "Moderate"),
    (28, 31, "Moderate"),
    (32, 34, "Severe"),
    (35, 36, "Severe"),
    (37, 46, "Very severe"),
]

# Function to classify stuttering events and calculate scores
def classify_stuttering_events(transcript):
    event_types = {}
    total_score = 0
    
    # Calculate severity score for each stuttering type based on frequency and weight
    for event, pattern in stuttering_types.items():
        matches = re.findall(pattern, transcript, re.IGNORECASE)
        event_count = len(matches)
        severity_score = event_count * severity_weights[event] if event_count > 0 else 0
        
        # Store the event severity score and level only if event is present
        if event_count > 0:
            event_types[event] = {
                "count": event_count,
                "severity_score": severity_score,
            }
            total_score += severity_score

    # Map the overall score to a severity level based on the SSI-4 table
    severity_level = "Very mild"  # Default to "Very mild" for low scores
    for min_score, max_score, level in severity_levels:
        if min_score <= total_score <= max_score:
            severity_level = level
            break

    # Calculate severity percentage based on maximum possible score
    max_possible_score = 46  # Based on the SSI-4 table's highest range
    severity_percentage = (total_score / max_possible_score) * 100

    return event_types, total_score, severity_percentage, severity_level

# Run classification and scoring for each transcript
for transcript in transcripts:
    event_types, total_score, severity_percentage, severity_level = classify_stuttering_events(transcript)
    print(f"Transcript: {transcript}")
    print("Detected Stuttering Types and Scores:")
    for event, data in event_types.items():
        print(f"  - {event.capitalize()}: Count = {data['count']}, Severity Score = {data['severity_score']}")
    print(f"Total Stuttering Score: {total_score}")
    print(f"Severity Percentage: {severity_percentage:.2f}%")
    print(f"Overall Severity Level: {severity_level}")
    print("-" * 50)

# Interface for PPO Agent
def ppo_agent_input(transcript):
    event_types, total_score, severity_percentage, severity_level = classify_stuttering_events(transcript)
    return {
        "transcript": transcript,
        "event_types": event_types,
        "total_score": total_score,
        "severity_percentage": severity_percentage,
        "severity_level": severity_level
    }

# Example usage for PPO input
ppo_input_example = ppo_agent_input("My-my name is Faaaaris Ziyad")
print("\nPPO Agent Input Example:")
print(ppo_input_example)
