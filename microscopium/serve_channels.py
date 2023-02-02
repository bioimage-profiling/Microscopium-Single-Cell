"""This module runs the bokeh server."""

import os
from os.path import dirname, join
from math import ceil, sqrt

import click
from skimage import io
from skimage.color import gray2rgba
import numpy as np
import pandas as pd

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models import (Button,
                          ColumnDataSource,
                          CustomJS,
                          CDSView,
                          GroupFilter,
                          Legend,
                          RadioButtonGroup,
                          Column)
from bokeh.events import ButtonClick
from bokeh.models.widgets import DataTable, TableColumn
import bokeh.palettes
from tornado import web

from .config import load_config, get_tooltips

def dataframe_from_file(filename, settings):
    """Read in pandas dataframe from filename."""
    df = pd.read_csv(filename, index_col=0)
    # columns = {}
    # for i in range(len(settings['image-columns'])):
    #     columns[settings['image-columns']] = f'channel{i}_path'
    # df.rename(columns=columns)
    return df

# change range to (0,1)
def change_range(image, C=0, D=255):
    A = image.min()
    B = image.max()
    image = (image - A)*(D - C)/(B - A) + C
    return image

# for image_rgb
def imread(paths):
    images = []
    for path in paths:
        image = io.imread(path).astype('float64')
        # change range to [0,1] 
        if image.ndim == 2: # gray image
            # image = change_range(image)
            image = image.astype('uint8')
            image = gray2rgba(image)
        elif image.ndim >= 3: # RGB image
            # for i in range(image.shape[-1]):
            #     image[:, :, i] = change_range(image[:, :, i])
            image = image.astype('uint8')
            shape = image.shape[:2]
            alpha = np.full((shape + (1,)), 255, dtype='uint8')
            image = np.concatenate((image, alpha), axis=2)
        else:
            exit('What kind of image is this!?!?!?!?')
        
        # for bokeh image_rgba function
        image = image.view("uint32").reshape(image.shape[:2])
        images.append(image)
        
    return images

def prepare_xy(source, settings):
    default_embedding = settings['embeddings']['default']
    embedding_x = settings['embeddings'][default_embedding]['x']
    embedding_y = settings['embeddings'][default_embedding]['y']
    source.add(source.data[embedding_x], name='x')
    source.add(source.data[embedding_y], name='y')


def source_from_dataframe(dataframe, settings, current_selection):
    """"""
    source = ColumnDataSource(dataframe)
    embeddings_names = list(settings['embeddings'].keys())
    selected_name = embeddings_names[current_selection]
    selected_column_x = settings['embeddings'][selected_name][0]
    selected_column_y = settings['embeddings'][selected_name][1]
    # Create empty columns to put selected coordinate data into
    source.add(dataframe[selected_column_x], name='x')
    source.add(dataframe[selected_column_y], name='y')
    return source


def update_image_canvas_single(index, data, source):
    """Update image canvas when a single image is selected on the scatter plot.

    The ``ColumnDataSource`` `source` will be updated in-place, which will
    cause the ``image_rgba`` plot to automatically update.

    Parameters
    ----------
    index : string
        The index value of the selected point. This must match the index on
        `data`.
    data : DataFrame
        The image properties dataset. It must include a 'path' pointing to the
        image file for each image.
    source : ColumnDataSource
        The ``image_rgba`` data source. It must include the columns 'image',
        'x', 'y', 'dx', 'dy'.
    """
    filename = data['AF750_imagepath'].iloc[index]
    # open images
    newimage = imread([filename])
    source.data = {'image': newimage, 'x': [0], 'y': [0], 'dx': [1], 'dy': [1]} # <== newimage is already list


def update_image_canvas_multi_channel(indices, data, source, settings):
    
    max_samples = len(settings['image-columns'])
    
    n_samples = len(indices)  
    if n_samples > max_samples:
        indices = indices[:max_samples]
        n_samples = len(indices)
    
    images = []
    for channelpath in settings['image-columns']:
        filenames = data.iloc[indices][channelpath]
        channelimages = imread(filenames)
        images += channelimages
      
    # Set sidelen to number of channels (NB! Only number of samples equal to the number of channels will be shown)
    sidelen =  len(settings['image-columns']) # 5x5 meshgrid
    step_size = 1 / sidelen # 0.2 stepsize
    grid_points_x = np.arange(0, 1 - step_size/2, step_size) # 5 grid points
    grid_points_y = np.arange(0, 1 - step_size/2, step_size) # 5 grid points
    start_xs, start_ys = np.meshgrid(grid_points_x, grid_points_y, indexing='ij') # start x, y for each grid
    n_rows = len(images) # number of rows that the images will occupy
    step_sizes = np.full(n_rows, step_size)
    margin = 0.05 * step_size / 2  
    source.data = {'image': images,
                   'x': start_xs.ravel() + margin,
                   'y': start_ys.ravel() + margin,
                   'dx': step_sizes * 0.95, 'dy': step_sizes * 0.95}
    
    # Set sidelen to number of channels (NB! Only number of samples equal to the number of channels will be shown)
    # sidelen_x = len(settings['image-columns']) # 5x5 meshgrid
    # sidelen_y = n_samples # 5x5 meshgrid
    # step_size_x = 1 / sidelen_x # stepsize for x axis
    # step_size_y = 1 / sidelen_y # stepsize for y axis
    # grid_points_x = np.arange(0, 1 - step_size_x/2, step_size_x) # 5 grid points
    # grid_points_y = np.arange(0, 1 - step_size_y/2, step_size_y) # 5 grid points
    # start_xs, start_ys = np.meshgrid(grid_points_x, grid_points_y, indexing='ij') # start x, y for each grid
    # # n_rows = len(images) # number of rows that the images will occupy
    # n_rows = len(images) # number of rows that the images will occupy
    # step_sizes_x = np.full(n_rows, step_size_x)
    # step_sizes_y = np.full(n_rows, step_size_y)
    # step_sizes_y[n_samples:] = 0
    # margin_x = 0.05 * step_size_x / 2
    # margin_y = 0.05 * step_size_y / 2
    # source.data = {'image': images,
    #                'x': start_xs.ravel() + margin_x,
    #                'y': start_ys.ravel() + margin_y,
    #                'dx': step_sizes_x * 0.95, 'dy': step_sizes_y * 0.95}


def _dynamic_range(fig, range_padding=0.05, range_padding_units='percent'):
    """Automatically rescales figure axes range when source data changes."""
    fig.x_range.range_padding = range_padding
    fig.x_range.range_padding_units = range_padding_units
    fig.y_range.range_padding = range_padding
    fig.y_range.range_padding_units = range_padding_units
    return fig


def _palette(num, type='categorical'):
    """Return a suitable palette for the given number of categories."""
    if type == 'categorical':
        if num in range(0, 3):
            return bokeh.palettes.Colorblind[3][:num]
        if num in range(3, 9):
            return bokeh.palettes.Colorblind[num]
        if num in range(9, 13):
            return bokeh.palettes.Set3[num]
        else:
            return bokeh.palettes.viridis(num)
    else:  # numerical
        return bokeh.palettes.viridis(num)


def embedding(source, settings):
    """Display a 2-dimensional embedding of the images.

    Parameters
    ----------
    source : ColumnDataSource
    settings : dictionary

    Returns
    -------
    embed : bokeh figure
        Scatterplot of precomputed x/y coordinates result
    """
    glyph_size = settings['plots']['glyph_size']
    tools_scatter = ['pan, box_select, poly_select, wheel_zoom, reset, tap']
    embed = figure(title='Embedding',
                   sizing_mode='scale_both',
                   tools=tools_scatter,
                   active_drag="box_select",
                   active_scroll='wheel_zoom',
                   tooltips=get_tooltips(settings),
                   output_backend='webgl')
    embed = _dynamic_range(embed)
    color_column = settings['color-columns']['categorical'][0]
    if color_column in source.data:
        group_names = pd.Series(source.data[color_column]).unique()
        my_colors = _palette(len(group_names))
        for i, group in enumerate(group_names):
            group_filter = GroupFilter(column_name=color_column, group=group)
            view = CDSView(source=source, filters=[group_filter])
            glyphs = embed.circle(x="x", y="y",
                                  source=source, view=view, size=glyph_size,
                                  color=my_colors[i], legend_label=group)
        embed.legend.location = "top_right"
        embed.legend.click_policy = "hide"
        embed.legend.background_fill_alpha = 0.5
    else:
        embed.circle(source=source, x='x', y='y', size=glyph_size)
    return embed


def _remove_axes_spines(plot):
    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None
    plot.xaxis.major_label_text_color = None
    plot.yaxis.major_label_text_color = None


def selected_images():
    """Create image canvas to display images from selected data.

    Returns
    -------
    selected_images : bokeh figure
    image_holder : data source to populate image figure
    """
    image_holder = ColumnDataSource({'image': [],
                                     'x': [], 'y': [],
                                     'dx': [], 'dy': []})
    tools_sel = ['pan, box_zoom, wheel_zoom, reset']
    selected_images = figure(title='Selected images',
                             x_range=[0, 1],
                             y_range=[0, 1],
                             sizing_mode='scale_both',
                             tools=tools_sel,
                             active_drag='pan',
                             active_scroll='wheel_zoom')
    # selected_images.image_url('image', 'x', 'y', 'dx', 'dy',
    #                           source=image_holder,
                            #   anchor='bottom_left')
    # selected_images.image('image', 'x', 'y', 'dx', 'dy',
    #                           source=image_holder,
    #                           # palette="Viridis256", 
    #                           )
    selected_images.image_rgba('image', 'x', 'y', 'dx', 'dy',
                              source=image_holder,
                              )
    _remove_axes_spines(selected_images)
    return selected_images, image_holder


def button_save_table(table):
    """Button to save selected data table as csv.

    Notes
    -----
    * Does not work for column values containing tuples (like 'neighbors')
    * Currently columns being saved are hard coded in the javascript callback
    * Available styles: 'default', 'primary', 'success', 'warning', 'danger'
    """
    button = Button(label="Download selected data", button_type="success")
    button.js_on_event(ButtonClick,
                       CustomJS(args=dict(source=table.source),
                                code=open(join(dirname(__file__), "js/download_data.js")).read()))

    return Column(button)


def button_print_page():
    """Button to print currently displayed webpage to paper or pdf.

    Notes
    -----
    * Available styles: 'default', 'primary', 'success', 'warning', 'danger'
    """
    button = Button(label="Print this page", button_type="success")
    button.js_on_event(ButtonClick, CustomJS(code="""print()"""))
    return Column(button)


def empty_table(df):
    """Display an empty table with column headings."""
    table_source = ColumnDataSource(pd.DataFrame(columns=df.columns))
    columns = [TableColumn(field=col, title=col) for col in df.columns]
    table = DataTable(source=table_source, columns=columns, width=800)
    return table


def update_table(indices, df, table):
    """Update table values to show only the currently selected data."""
    filtered_df = df.iloc[indices]
    table.source.data = dict(ColumnDataSource(filtered_df).data)


def switch_embeddings_button_group(settings):
    """Create radio button group for switching between UMAP, tSNE, and PCA."""
    default_embedding = settings['embeddings']['default']
    del settings['embeddings']['default']
    button_labels = list(settings['embeddings'].keys())
    default_embedding_idx = button_labels.index(default_embedding)
    radio_button_group = RadioButtonGroup(labels=button_labels,
                                          active=default_embedding_idx)
    return radio_button_group


def update_embedding(source, embedding, settings):
    """Update source of image embedding scatterplot."""
    embeddings = settings['embeddings']
    x_source = embeddings[embedding]['x']
    y_source = embeddings[embedding]['y']
    source.data['x'] = source.data[x_source]
    source.data['y'] = source.data[y_source]
    source.trigger("data", 0, 0)


def reset_plot_axes(plot, x_start=0, x_end=1, y_start=0, y_end=1):
    plot.x_range.start = x_start
    plot.x_range.end = x_end
    plot.y_range.start = y_start
    plot.y_range.end = y_end


def make_makedoc(filename, settings_filename):
    """Make the makedoc function required by Bokeh Server.

    To run a Bokeh server, we need to create a function that takes in a Bokeh
    "document" as input, and adds our figure (together with all the interactive
    bells and whistles we may want to add to it) to that document. This then
    initialises a ``FunctionHandler`` and an ``Application`` gets started by
    the server. See the `run_server` code for details.

    Parameters
    ----------
    filename : string
        A CSV file containing the data for the app.
    settings_filename: string
        Path to a yaml file

    Returns
    -------
    makedoc : function
        A makedoc function as expected by ``FunctionHandler``.
    """
    settings = load_config(settings_filename)
    dataframe = dataframe_from_file(filename, settings)

    def makedoc(doc):
        source = ColumnDataSource(dataframe) # columndatasource for 'only' plotting umap
        prepare_xy(source, settings)  # get the default embedding columns
        embed = embedding(source, settings)
        image_plot, image_holder = selected_images() # image_holder: columndatasource for only images
        table = empty_table(dataframe)
        controls = [button_save_table(table), button_print_page()]
        radio_buttons = switch_embeddings_button_group(settings)

        def load_selected(attr, old, new):
            """Update images and table to display selected data."""
            print('new index: ', new)
            # Update images & table
            if len(new) == 1:  # could be empty selection
                update_image_canvas_single(new[0], data=dataframe, source=image_holder)
                # update_image_canvas_multi_channel(new[0], data=dataframe, source=image_holder)
            elif len(new) > 1:
                # update_image_canvas_multi(new, data=dataframe, source=image_holder)
                update_image_canvas_multi_channel(new, data=dataframe, source=image_holder, settings=settings)
            
            reset_plot_axes(image_plot)  # effectively resets zoom level
            update_table(new, dataframe, table)

        def new_embedding(attr, old, new):
            embedding = list(settings['embeddings'])[radio_buttons.active]
            update_embedding(source, embedding, settings)

        source.selected.on_change('indices', load_selected)
        radio_buttons.on_change('active', new_embedding)

        page_content = layout([
            radio_buttons,
            [embed, image_plot],
            controls,
            [table]
        ], sizing_mode="scale_width")
        doc.title = 'Bokeh microscopium app'
        doc.add_root(page_content)
    print('ready!')
    return makedoc


def default_config(filename):
    d = os.path.dirname(filename)
    return os.path.join(d, 'settings.yaml')


@click.command()
@click.argument('filename')
@click.option('-c', '--config', default=None)
@click.option('-p', '--path', default='/')
@click.option('-P', '--port', type=int, default=5000)
@click.option('-u', '--url', default='http://localhost')
def run_server_cmd(filename, config=None, path='/', port=5000,
                   url='http://localhost'):
    run_server(filename, config=config, path=path, port=port, url=url)


def run_server(filename, config=None, path='/', port=5000, url='http://localhost'):
    """Run the bokeh server."""
    if config is None:
        config = default_config(filename)
    makedoc = make_makedoc(filename, config)
    apps = {path: Application(FunctionHandler(makedoc))}
    server = Server(apps, port=port, allow_websocket_origin=['*'])
    server.start()
    print('Web app now available at {}:{}'.format(url, port))
    handlers = [(path + r'images/(.*)',
                 web.StaticFileHandler,
                 {'path': os.path.dirname(filename)})]
    server._tornado.add_handlers(r".*", handlers)
    server.run_until_shutdown()


if __name__ == '__main__':
    run_server_cmd()
