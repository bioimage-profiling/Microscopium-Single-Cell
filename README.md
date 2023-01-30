- You will need <strong>settings.yaml</strong> and <strong>a csv file</strong> with umap coordinates and metadata.

- In the csv file columns include at least:
	- umap coordinates (e.g., umap_x, umap_y)
	- nucleus center coordinates (e.g., centroid_x, centroid_y)
	- image paths (e.g., path)

<br />

- In the settings.yaml configures which column names from the csv file to use.
	- <strong>embeddings:</strong> column names in the csv file to use as umap coordinates.
	- <strong>centroid-x-column:</strong> x coordinate column for nucleus center
	- <strong>centroid-y-column:</strong> y coordinate column for nucleus center
	- <strong>color-columns:</strong> column to be used as legend
	- <strong>image-column:</strong> column that contains path to image files
	- <strong>tooltip-columns:</strong> columns to show when hovering on the umap
	- <strong>glyph_size:</strong> size of the dots int the umap

<br />

- Run with <code style="color:blue">python -m microscopium.serve <em style="color:black">path to csv file</em> -c <em style="color:black">path to setting.yaml file</em></code>
- For example: <code style="color:blue">python -m microscopium.serve metafile/umap_coords.csv\
        -c metafile/settings.yaml </code>

- Lastly open <link>http://localhost:5000</link>


### Requirements

Common python packages + <code style="color:blue">skimage, pandas, bokeh, tornado</code>