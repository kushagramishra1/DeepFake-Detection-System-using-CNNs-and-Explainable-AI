from models.explainability import explain_prediction
import matplotlib.pyplot as plt

def main(image_path):
    result = explain_prediction(image_path)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")

    # Display heatmap
    plt.imshow(result['heatmap'], cmap='jet')
    plt.title('Explanation Heatmap')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])