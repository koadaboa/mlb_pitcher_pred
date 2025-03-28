# src/scripts/model_diagnostic.py
import pickle
import sys
import argparse

def analyze_model(model_path):
    """Analyze a trained model and print feature information"""
    print(f"Analyzing model: {model_path}")
    
    # Load the model
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Print basic model info
    print(f"\nModel type: {type(model_dict['model']).__name__}")
    print(f"Performance metrics: {model_dict['metrics']}")
    
    # Print the features used
    print("\nFeatures used in the model:")
    print(model_dict['features'])
    
    # Print feature importances
    feature_importances = list(zip(model_dict['features'], model_dict['model'].feature_importances_))
    sorted_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    
    print("\nFeature importances:")
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")
    
    # Print total importance
    print(f"\nTotal importance sum: {sum(imp for _, imp in sorted_importances):.4f}")
    
    # Check which new features were actually included
    enhanced_feature_patterns = ['_std', 'trend_', 'momentum_', 'entropy', 'pitch_pct_']
    enhanced_features = [f for f in model_dict['features'] if any(pattern in f for pattern in enhanced_feature_patterns)]
    
    if enhanced_features:
        print("\nEnhanced features included in the model:")
        for feat in enhanced_features:
            idx = model_dict['features'].index(feat)
            imp = model_dict['model'].feature_importances_[idx]
            print(f"{feat}: {imp:.4f}")
    else:
        print("\nNO ENHANCED FEATURES FOUND IN THE MODEL")
        print("This suggests the enhanced features weren't properly calculated or included in feature selection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a trained model')
    parser.add_argument('model_path', nargs='?', default='models/strikeout_model.pkl', 
                      help='Path to the trained model pickle file')
    args = parser.parse_args()
    
    analyze_model(args.model_path)