import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from src.base_tags import load_base_tags
from src.data_fetch import load_data
from src.vector import generate_vector
import ast


def main():
    # Load the main data points
    points_df = load_data()
    # Load the base tags and their associated scores/embeddings
    base_tags_df = load_base_tags()

    # Store the calculated scores for each row
    scores_list = []

    # Iterate over each data point
    for _, row in points_df.iterrows():
        tags = ast.literal_eval(row["tags"])
        # Initialize score dictionary for each category
        scores = {
            "education_score": 0,
            "tourism_score": 0,
            "welfare_score": 0,
            "terrain_score": 0,
            "transport_score": 0,
            "residential_score": 0,
            "commercial_score": 0,
            "base_demand_score": 0,
            "morning_peak_factor": 0,
            "evening_peak_factor": 0,
            "daytime_factor": 0,
            "weekend_factor": 0,
            "weather_sensitivity": 0,
            "seasonal_variation": 0,
            "stop_type": "",
        }

        # Store hit_tags for this data point
        hit_tags = []

        # Process each tag for the current data point
        for tag in tags:
            # If the tag is found in the base tags, add its scores directly
            if tag in base_tags_df["name"].values:
                for key in scores.keys():
                    if key != "stop_type":  # Skip non-numeric columns
                        scores[key] += base_tags_df.loc[
                            base_tags_df["name"] == tag, key
                        ].values[0]
                    else:
                        hit_tags.append(base_tags_df.loc[
                            base_tags_df["name"] == tag, key
                        ].values[0])
            else:
                # If the tag is not found, generate its vector and find the 5 most similar base tags
                tag_vector = generate_vector(tag)
                similarities = base_tags_df["embedding"].apply(
                    lambda x: cosine_similarity(tag_vector.reshape(1, -1), x.reshape(1, -1))[0][0]
                )
                best_5_indices = similarities.nlargest(5).index
                best_5_tags = base_tags_df.loc[best_5_indices]
                best_5_similarities = similarities.loc[best_5_indices]
                
                # Sort by similarity in descending order to maintain ranking
                similarity_sort_order = best_5_similarities.sort_values(ascending=False).index
                best_5_tags = best_5_tags.loc[similarity_sort_order]
                best_5_similarities = best_5_similarities.loc[similarity_sort_order]
                similarity_sum = best_5_similarities.sum()

                # Update hit_tags
                hit_tags.append(best_5_tags["stop_type"].values[0])

                # Check the its validation
                # print(f"Tag {tag} is similar to {best_5_tags["name"].values}")
                
                # Assign stop_type from the most similar tag
                most_similar_idx = best_5_similarities.idxmax()
                scores["stop_type"] = base_tags_df.loc[most_similar_idx, "stop_type"]
                
                # Weighted sum of the top 5 similar tags' scores
                for _, best_row in best_5_tags.iterrows():
                    similarity = best_5_similarities[best_row.name]
                    weight = similarity / similarity_sum
                    for key in scores.keys():
                        if key != "stop_type":  # Skip non-numeric columns
                            scores[key] += best_row[key] * weight
            
        # Find the most common stop_type in hit_tags
        if hit_tags:
            # Flatten the hit_tags list (it may contain arrays) and convert to strings
            flattened_tags = []
            for tag in hit_tags:
                if isinstance(tag, (list, tuple)):
                    flattened_tags.extend([str(t) for t in tag])
                else:
                    flattened_tags.append(str(tag))
            
            # Get the most common stop_type
            if flattened_tags:
                counter = Counter(flattened_tags)
                most_common_stop_type = counter.most_common(1)[0][0]
                scores["stop_type"] = most_common_stop_type
        
        # Average the scores by the number of tags
        tags_amount = len(tags)
        for key in scores.keys():
            if key != "stop_type":  # Skip non-numeric columns
                scores[key] = scores[key] / tags_amount
        
        # Store the scores for this row
        scores_list.append(scores)
    
    # Convert the scores list to a DataFrame and add them as new columns
    scores_df = pd.DataFrame(scores_list)
    points_df = pd.concat([points_df, scores_df], axis=1)
    
    # Display the updated DataFrame
    points_df.to_csv("./data/expanded_points.csv", index=False)


if __name__ == "__main__":
    main()
