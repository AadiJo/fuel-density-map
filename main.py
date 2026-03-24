import os
import analysis
import numpy as np
from PIL import Image

def raw_data_analysis():
    # Example usage
    vid_path = "videos/citrus_bread_wr_blackboxed.mp4"
    output_dir = "output"

    start_time = 0  # in seconds
    end_time = None   # in seconds
    
    print("Analyzing video...")
    vid_totals = analysis.get_total_yellow_pixels_from_video(vid_path, start_time, end_time)
    # Save the raw results to a file

    print("Saving results...")
    raw_data_file_name = vid_path.split("/")[-1].split(".")[0] + "_raw_data.txt"
    with open(os.path.join(output_dir, raw_data_file_name), "w") as f:
        for row in vid_totals:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    print("Results saved.")

def process_raw_data():
    data_path = "output/citrus_bread_wr_blackboxed_raw_data.txt"
    processed_image_path = "images"
    pct_from_average_to_max = 0.5 # how much of the way from the average to the max value should the average color be, as a percentage
    average_display_color = (255, 0, 255) # magenta, easy to see

    print("Processing raw data...")
    # Read the raw data from the file
    raw_data = np.loadtxt(data_path, dtype=np.int32)
    
    processed_folder_name = data_path.split("/")[-1].split("_")[0] + "_processed"
    processed_folder_path = os.path.join(processed_image_path, processed_folder_name)
    os.makedirs(processed_folder_path, exist_ok=True)

    max_value = int(raw_data.max())
    non_zero_values = raw_data[raw_data > 0]
    actual_average = float(non_zero_values.mean()) if non_zero_values.size else 0.0
    average_of_non_zero_values = actual_average + pct_from_average_to_max * (max_value - actual_average)
    print(f"Max value: {max_value}")
    print(f"Actual average of non-zero values: {actual_average}")
    print(f"Used Average of non-zero values: {average_of_non_zero_values}")

    # turn the raw data into an array of colors where the color is determined by the value of the raw data at that point, 
    # with 0 being black, 
    # max_value being white,
    #  average_of_non_zero_values being average_display_color
    # and values between 0 and average_of_non_zero_values being a gradient from black to average_display_color,
    # and values between average_of_non_zero_values and max_value being a gradient from average_display_color to white

    color_array = create_color_array(raw_data, max_value, average_of_non_zero_values, average_display_color)

    print("Creating image...")
    img = Image.fromarray(color_array, mode="RGB")
    img.save(os.path.join(processed_folder_path, "processed_image.png"))
    print("Image saved.")

    print("Creating transparent image...")
    alpha_channel = np.where(np.any(color_array != 0, axis=2), 128, 0).astype(np.uint8)
    transparent_image = np.dstack((color_array, alpha_channel))
    img = Image.fromarray(transparent_image, mode="RGBA")
    img.save(os.path.join(processed_folder_path, "processed_image_transparent.png"))
    print("Transparent image saved.")

def create_color_array(raw_data, max_value, average_of_non_zero_values, average_display_color):
    color_array = np.zeros((*raw_data.shape, 3), dtype=np.uint8)

    if max_value <= 0:
        return color_array

    average_color = np.array(average_display_color, dtype=np.float32)
    non_zero_mask = raw_data > 0

    if average_of_non_zero_values > 0:
        lower_mask = non_zero_mask & (raw_data <= average_of_non_zero_values)
        if np.any(lower_mask):
            lower_ratio = (raw_data[lower_mask] / average_of_non_zero_values).astype(np.float32)
            color_array[lower_mask] = np.clip(lower_ratio[:, None] * average_color, 0, 255).astype(np.uint8)

    upper_mask = non_zero_mask & (raw_data > average_of_non_zero_values)
    upper_range = max_value - average_of_non_zero_values
    if np.any(upper_mask):
        if upper_range <= 0:
            color_array[upper_mask] = 255
        else:
            upper_ratio = ((raw_data[upper_mask] - average_of_non_zero_values) / upper_range).astype(np.float32)
            color_array[upper_mask] = np.clip(
                average_color + upper_ratio[:, None] * (255 - average_color),
                0,
                255,
            ).astype(np.uint8)

    color_array[raw_data >= max_value] = 255
    return color_array


if __name__ == "__main__":
    raw_data_analysis()
    process_raw_data()
