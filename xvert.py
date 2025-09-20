import imageio  # Already installed in your venv

# Convert GIF to MP4
input_file = "m_epicycles.gif"
output_file = "m_epicycles.mp4"

with imageio.get_reader(input_file) as reader:
    with imageio.get_writer(output_file, fps=10) as writer:  # fps=10 matches our animation
        for frame in reader:
            writer.append_data(frame)

print(f"Converted! Upload '{output_file}' to X.")