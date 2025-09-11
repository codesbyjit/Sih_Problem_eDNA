def classify_clusters(representative_sequences, model):
    """Predicts taxonomy using the trained model and its confidence."""
    print("Classifying representative sequences...")
    
    # Get predictions and the probabilities for each class
    predictions = model.predict(representative_sequences)
    probabilities = model.predict_proba(representative_sequences)
    
    # Get the confidence score for the winning class
    confidence_scores = probabilities.max(axis=1)
    
    # If confidence is low, it might be a novel or unclassified organism
    final_predictions = []
    for pred, conf in zip(predictions, confidence_scores):
        if conf < 0.75: # Confidence threshold can be tuned
            final_predictions.append(f"Unclassified (Confidence: {conf:.2f})")
        else:
            final_predictions.append(pred)
            
    return final_predictions