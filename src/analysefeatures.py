from dataframe import df
import matplotlib.pyplot as plt

feature_vector = ["area", "perimeter", "vertices", "aspect_ratio", "circularity"]


for feature in feature_vector:
                plt.figure()
                print("Available columns:", df.columns.tolist())
                plt.hist(df[feature], bins=15)
                plt.xlabel(feature)
                plt.ylabel("Frequency")
                plt.title(f"Distribution of {feature}")
                for label in df["label"].unique():
                    values = df[df["label"] == label][feature]
                    plt.hist(values, alpha=0.5, label=label)
                output_filename = f"{feature}_histogram.png"
                plt.savefig(output_filename, bbox_inches='tight')
                print(f"📈 Plot successfully saved to {output_filename}")

                
                # Clear the current figure so the next loop/plot doesn't overlap
                plt.clf() 