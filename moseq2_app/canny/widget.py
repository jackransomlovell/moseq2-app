'''
Constructs a jupyter notebook viewable widget users can use to identify the arena floor
and to validate extractions performed on a small chunk of data. 
'''

import param
import numpy as np
import panel as pn
import holoviews as hv
from operator import add
from copy import deepcopy
from tqdm.auto import tqdm
from functools import reduce
from panel.viewable import Viewer
from bokeh.models import HoverTool
from os.path import exists, basename, dirname
from moseq2_app.gui.progress import get_sessions
from moseq2_extract.io.video import load_movie_data
from moseq2_extract.util import select_strel, get_strels
from moseq2_extract.extract.extract import extract_chunk
from moseq2_extract.util import detect_and_set_camera_parameters
from moseq2_app.util import write_yaml, read_yaml, read_and_clean_config, update_config
from moseq2_extract.extract.proc import threshold_chunk, get_canny_msk, get_bground_im_file



# contains display parameters for the bokeh hover tools
_hover_dict = {
    'Background': HoverTool(tooltips=[('Distance from camera (mm)', '@image')]),
    'Extracted mouse': HoverTool(tooltips=[('Height from floor (mm)', '@image')]),
    'Frame (background subtracted)': HoverTool(tooltips=[('Height from floor (mm)', '@image')]),
}


class CannyWidget:
    def __init__(self, data_dir, config_file, session_config_path, skip_extracted=False) -> None:
        self.backgrounds = {}
        self.extracts = {}
        self.data_dir = data_dir
        self.session_config_path = session_config_path

        self.config_data = read_and_clean_config(config_file)

        sessions = get_sessions(data_dir, skip_extracted=skip_extracted)
        if len(sessions) == 0:
            self.view = None
            print('No sessions to show. There are either no sessions present or they are all extracted.')
            return

        # creates session-specific configurations
        if exists(session_config_path):
            session_parameters = read_yaml(session_config_path)
            new_sessions = set(map(lambda f: basename(dirname(f)), sessions)) - set(session_parameters)
            if len(new_sessions) > 0:
                for _session in tqdm(new_sessions, desc="Setting camera parameters", leave=False):
                    full_path = [x for x in sessions if _session in x][0]
                    session_parameters[_session] = detect_and_set_camera_parameters(
                        deepcopy(self.config_data), full_path)
                # write session config with default parameters for new sessions
                write_yaml(session_parameters, self.session_config_path)

        else:
            session_parameters = {basename(dirname(f)): detect_and_set_camera_parameters(
                deepcopy(self.config_data), f) for f in tqdm(sessions, desc="Setting camera parameters", leave=False)}
            write_yaml(session_parameters, self.session_config_path)

        # add session_config path to config.yaml

        with update_config(config_file) as temp_config_data:
            temp_config_data['session_config_path'] = session_config_path

        self.session_config = session_parameters
        self.sessions = {basename(dirname(f)): f for f in sessions}

        # instantiate CannyData with the first session in this list
        self.session_data = CannyData(path=list(self.sessions)[0], controller=self)
        self.session_data.param.path.objects = list(self.sessions)  # generates object selector in gui

        self.view = CannyView(self.session_data)

    def _repr_mimebundle_(self, include=None, exclude=None):
        if self.view is not None:
            return self.view._repr_mimebundle_(include, exclude)


    def set_session_config_vars(self):
        folder = self.session_data.path
        session = self.session_config[folder]

        # canny thresholds
        session['t1'] = self.session_data.t1
        session['t2'] = self.session_data.t2

        session['tail_filter_iters'] = self.session_data.tail_filters
        session['tail_filter_shape'] = self.session_data.tail_filter_shape
        session['tail_filter_size'] = (self.session_data.tail_filter_size, ) * 2

        session['otsu'] = self.session_data.otsu

        session['final_dilate'] = self.session_data.final_dilate

        session['min_height'] = self.session_data.min_height
        session['max_height'] = self.session_data.max_height


    def save_session_parameters(self):
        self.set_session_config_vars()

        write_yaml(self.session_config, self.session_config_path)

    def compute_extraction(self):
        self.set_session_config_vars()

        folder = self.session_data.path
        background = self.get_background(folder)
        session_config = self.session_config[folder]

        # get segmented frame
        raw_frames = load_movie_data(self.sessions[folder],
                                     range(1, self.session_data.frames_to_extract + 1),
                                     **session_config,
                                     frame_size=background.shape[::-1])

        # subtract background
        frames = (background - raw_frames)

        # filter for included mouse height range
        global_roi = self.session_data.images['Global ROI']
        frames = threshold_chunk(frames, session_config['min_height'], session_config['max_height'])*global_roi

        wall_roi = self.session_data.images['Wall ROI']
        floor_roi = (~wall_roi.astype(bool)).astype(np.uint8)

        msks = []
        for i in range(frames.shape[0]):
            msk = get_canny_msk(frames,
                                wall_roi,
                                floor_roi, 
                                session_config['t1'], session_config['t2'],
                                tail_size=session_config['tail_filter_size'], 
                                otsu=session_config['otsu'],
                                final_dilate=session_config['final_dilate'])
            msks.append(msk)
            frames[i] = frames[i]*msk
        
        extraction = extract_chunk(frames)
        
        return extraction['depth_frames'], frames

    def get_background(self, folder=None):
        '''Assuming this will be called by an event-triggered function'''
        if folder is None:
            folder = self.session_data.path
        if folder not in self.backgrounds:
            # get full path
            file = self.sessions[folder]
            # compute background
            bground = get_bground_im_file(file, frame_stride=1000)
            # save for recall later
            self.backgrounds[folder] = bground

        return self.backgrounds[folder]
    
# define data class first
class CannyData(param.Parameterized):
    '''The link between the widget view and the underlying data'''
    ### data ###
    path = param.ObjectSelector()  # stores the currently selected path and all others
    # defines thresholds used to locate mouse for extractions
    mouse_height = param.Range(default=(10, 300), bounds=(0, 500), label="Rat height clip (mm)")
    # stores the extracted frame number to display and how many to extract in test
    frame_num = param.Integer(default=0, bounds=(0, 99), label="Display frame (index)")
    frames_to_extract = param.Integer(default=100, bounds=(1, None), label="Number of test frames to extract")
    # stores images of the arena and extractions
    images = param.Dict(default={'Background': None, 'Wall ROI': None, 'Global ROI': None,
                        'Extracted mouse': None, 'Frame (background subtracted)': None})
    # stores class object that holds the underlying data
    controller: CannyWidget = param.Parameter()

    ### advanced extraction parameters ###
    adv_extraction_flag = param.Boolean(label="Show advanced extraction parameters")
    crop_size = param.Integer(default=80, bounds=(1, None), label="Crop size (width and height; pixels)")
    flip_classifier = param.Selector(label='Flip classifier')  # TODO: fill this one out better
    flip_classifier_smoothing = param.Integer(default=51, bounds=(1, None), label="Flip classifier smoothing (frames)")

    tracking_model_flag = param.Boolean(default=False, label="Use tracking model")
    tracking_model_mask_thresh = param.Number(default=-16, bounds=(None, 0), label="Tracking model mouse likelihood threshold")

    cable_filters = param.Integer(default=0, bounds=(0, None), label="Cable filter iterations")
    cable_filter_shape = param.Selector(objects=['rectangle', 'ellipse'], default='rectangle', label="Cable filter shape")
    cable_filter_size = param.Integer(default=5, bounds=(3, None), step=2, label="Cable filter size")

    tail_filters = param.Integer(default=1, bounds=(0, None), label="Tail filter iterations")
    tail_filter_shape = param.Selector(objects=['rectangle', 'ellipse'], default='ellipse', label="Tail filter shape")
    tail_filter_size = param.Integer(default=9, bounds=(3, None), step=2, label="Tail filter size")

    spatial_filter = param.String(default="[3]", label="Spatial filter(s)")
    temporal_filter = param.String(default="[0]", label="Temporal filter(s)")

    chunk_overlap = param.Integer(default=0, bounds=(0, None), label="Extraction chunk overlap (useful for tracking model)")
    frame_dtype = param.Selector(default="uint16", objects=["uint8", "uint16"], label="Output extraction datatype")
    # NOTE: in the past this has been <i2, so might change some people's analysis
    movie_dtype = param.String(default="<u2", label="Raw depth video datatype (for .dat files)")
    pixel_format = param.Selector(default="gray16le", objects=["gray16le", "gray16be"], label="Raw depth video datatype (for .avi or .mkv files)")
    compress = param.Boolean(default=False, label="Compress raw data after extraction")

    ### flags and actions ###
    computing_extraction = param.Boolean(default=False)  # used to indicate a computation is being performed
    # used to trigger the computation of a small extraction via a button
    compute_extraction = param.Action(lambda x: x.param.trigger('compute_extraction'), label="Compute extraction")
    save_session_and_move_btn = param.Action(lambda x: x.param.trigger('save_session_and_move_btn'), label="Save session parameters and move to next")
    save_session_btn = param.Action(lambda x: x.param.trigger('save_session_btn'), label="Save session parameters")
    load_rois = param.Action(lambda x: x.param.trigger('load_rois'), label="Load backgrounds for ROI drawing")
    draw_rois = param.Action(lambda x: x.param.trigger('draw_rois'), label="Draw ROI")

    ### canny algo params
    canny_t1 = param.Integer(default=100, bounds=(0, 255), label="Canny t1")
    canny_t2 = param.Integer(default=200, bounds=(0, 255), label="Canny t2")
    otsu = param.Boolean(default=False, label="Use Otsu thresholding")
    final_dilate = param.Integer(default=0, bounds=(0, None), label="Final dilation iterations")

    @param.depends('save_session_btn', 'save_session_and_move_btn', watch=True)
    def save_session(self):
        self.controller.save_session_parameters()

    @param.depends('save_session_and_move_btn', watch=True)
    def next_session(self):
        cur_path = self.path
        paths = self.param.path.objects
        try:
            next_path_index = (paths.index(cur_path) + 1) % len(paths)
            self.path = paths[next_path_index]
        except ValueError:
            self.path = paths[0]

    @param.depends('compute_extraction', 'next_session', watch=True)
    def get_extraction(self):
        self.computing_extraction = True
        mouse, frame = self.controller.compute_extraction()
        self.images['Extracted mouse'] = mouse
        self.images['Frame (background subtracted)'] = frame

        self.computing_extraction = False

    @param.depends('get_extraction', watch=True)
    def change_frame_slider(self):
        self.param.frame_num.bounds = (0, min(self.frames_to_extract - 1, len(self.images['Extracted mouse']) - 1))

    @param.depends('load_rois', watch=True)
    def load_roi(self):
        bground = self.controller.get_background()
        self.images['Global ROI'] = bground
        self.images['Wall ROI'] = bground

    @param.depends('draw_rois', watch=True)
    def draw_roi(self):

        for k,v in self.images():
            if k not in ['Global ROI', 'Wall ROI']:
                continue

            bground = v
            x=(int(min(self.poly_stream[k].data['xs'][0])),int(max(self.poly_stream[k].data['xs'][0])))
            tmp=(int(min(self.poly_stream[k].data['ys'][0])),int(max(self.poly_stream[k].data['ys'][0])))
            y=(bground.shape[0]-tmp[1],bground.shape[0]-tmp[0])

            roi = np.zeros(bground.shape, dtype=np.uint8)
            roi[y[0]:y[1], x[0]:x[1]] = 1

            self.images[k] = roi
        
    @param.depends('get_extraction', 'frame_num', 'load_rois', 'draw_rois')
    def display(self):
        panels = []
        for k, v in self.images.items():
            if not isinstance(v, (np.ndarray, list)):
                im = hv.Image([], label=k)
            elif k in ('Extracted mouse', 'Frame (background subtracted)'):
                _img = v[min(len(v) - 1, self.frame_num)]
                im = hv.Image(_img, label=k, bounds=(0, 0, v.shape[2], v.shape[1])
                              ).opts(xlim=(0, v.shape[2]), ylim=(0, v.shape[1]), framewise=True)
            elif k in ('Global ROI', 'Wall ROI'):
                bground = v
                vmin = 750
                vmax = bground.max()

                img = hv.Image(bground, 
                               bounds = (0, bground.shape[0], bground.shape[1], 0)).opts(width=512,
                                                                                         height=424,
                                                                                         clim=(vmin, vmax))
                poly = hv.Polygons([])
                self.poly_stream[k] = hv.streams.PolyDraw(source=poly, drag=True, num_objects=4,
                                                          show_vertices=True, styles={
                                                              'fill_color': ['red', 'green']
                                                              })

                im = (img * poly).opts(
                    hv.opts.Path(color='red', height=400, line_width=5, width=400),
                    hv.opts.Polygons(fill_alpha=0.3, active_tools=['poly_draw']))
            else:
                im = hv.Image(v, label=k, bounds=(
                    0, 0, *v.shape[::-1])).opts(xlim=(0, v.shape[1]), ylim=(0, v.shape[0]), framewise=True)
            panels.append(im.opts(
                hv.opts.Image(
                    tools=[_hover_dict[k]],
                    cmap='cubehelix',
                    xlabel="Width (pixels)",
                    ylabel="Height (pixels)",
                ),
            ))
        panels = reduce(add, panels).opts(hv.opts.Image()).cols(2)
        return panels.opts(shared_axes=False, framewise=True)


class CannyView(Viewer):

    def __init__(self, session_data: CannyData, **params):
        super().__init__(**params)

        def _link_data(widget, param, **kwargs):
            return widget.from_param(getattr(session_data.param, param), **kwargs)

        # define widget column

        ### subsection: session selector ###
        session_selector = _link_data(pn.widgets.Select, "path", size=4, name="")

        ### subsection: extraction parameters ###
        clip_mouse_height = _link_data(pn.widgets.IntRangeSlider, "mouse_height")
        display_frame_num = _link_data(pn.widgets.IntSlider, "frame_num")
        extraction_frame_num = _link_data(pn.widgets.IntInput, "frames_to_extract")
        compute_extraction_btn = _link_data(pn.widgets.Button, "compute_extraction")
        show_adv_extraction = _link_data(pn.widgets.Checkbox, "adv_extraction_flag")

        ### subsection: optional advanced extraction parameters ###
        crop_size = _link_data(pn.widgets.NumberInput, "crop_size", height=40)
        # TODO: add flip classifier
        flip_classifier_smoothing = _link_data(pn.widgets.NumberInput, "flip_classifier_smoothing", height=40)
        tracking_model_flag = _link_data(pn.widgets.Checkbox, "tracking_model_flag", height=20)
        tracking_model_mask_thresh = _link_data(pn.widgets.NumberInput, "tracking_model_mask_thresh", height=40)

        cable_filters = _link_data(pn.widgets.IntInput, "cable_filters", height=40)
        cable_filter_shape = _link_data(pn.widgets.Select, "cable_filter_shape", height=40)
        cable_filter_size = _link_data(pn.widgets.IntInput, "cable_filter_size", height=50)

        tail_filters = _link_data(pn.widgets.IntInput, "tail_filters", height=40)
        tail_filter_shape = _link_data(pn.widgets.Select, "tail_filter_shape", height=40)
        tail_filter_size = _link_data(pn.widgets.IntInput, "tail_filter_size", height=40)

        spatial_filter = _link_data(pn.widgets.TextInput, "spatial_filter", height=40)
        temporal_filter = _link_data(pn.widgets.TextInput, "temporal_filter", height=40)

        chunk_overlap = _link_data(pn.widgets.IntInput, "chunk_overlap", height=40)
        frame_dtype = _link_data(pn.widgets.Select, "frame_dtype", height=40)
        movie_dtype = _link_data(pn.widgets.TextInput, "movie_dtype", height=40)
        pixel_format = _link_data(pn.widgets.Select, "pixel_format", height=40)
        compress = _link_data(pn.widgets.Checkbox, "compress", height=15)

        ### subsection: ROI selection ###
        canny_t1 = _link_data(pn.widgets.IntInput, "canny_t1", height=40)
        canny_t2 = _link_data(pn.widgets.IntInput, "canny_t2", height=40)
        otsu = _link_data(pn.widgets.Checkbox, "otsu", height=15)
        final_dilation = _link_data(pn.widgets.IntInput, "final_dilation", height=40)
        load_rois_btn = _link_data(pn.widgets.Button, "load_rois")
        draw_rois_btn = _link_data(pn.widgets.Button, "draw_rois")

        adv_extraction = pn.Column(
            crop_size,
            flip_classifier_smoothing,
            pn.pane.Markdown("#### Tail filtering parameters", height=20, width=300),
            tail_filters,
            tail_filter_shape,
            tail_filter_size,
            pn.pane.Markdown("#### Pose filtering parameters", height=20, width=300),
            spatial_filter,
            temporal_filter,
            pn.pane.Markdown("#### Video parameters", height=20, width=300),
            chunk_overlap,
            frame_dtype,
            movie_dtype,
            pixel_format,
            compress,
            pn.pane.Markdown("#### Tracking model parameters", height=20, width=300),
            tracking_model_flag,
            tracking_model_mask_thresh,
            pn.pane.Markdown("#### Cable filtering parameters", height=20, width=300),
            cable_filters,
            cable_filter_shape,
            cable_filter_size,
            pn.pane.Markdown("#### Canny algo parameters", height=20, width=300),
            canny_t1,
            canny_t2,
            otsu,
            final_dilation,
            visible=False,
            scroll=True,
            height=200
            )
        show_adv_extraction.link(adv_extraction, value='visible')

        ### subsection: saving and indicator ###
        save_session_and_move_btn = _link_data(pn.widgets.Button, "save_session_and_move_btn")
        save_session_btn = _link_data(pn.widgets.Button, "save_session_btn")

        def _link_button_visibility(target, event):
            target.visible = not event.new

        # computing extraction
        indicator2 = pn.Row(
            pn.pane.Markdown('Extracting...', width=100),
            pn.widgets.Progress(active=True, bar_color='info', width=160),
            visible=False,
            height=45,
        )
        computing_check2 = _link_data(pn.widgets.Checkbox, "computing_extraction", value=False, visible=False)
        computing_check2.link(indicator2, value='visible')
        computing_check2.link(compute_extraction_btn, callbacks={'value': _link_button_visibility})

        ### link all subsections into GUI layout ###

        # combine widgets
        self.gui_col = pn.Column(
            '### Sessions',
            session_selector,
            '### Extraction parameters',
            clip_mouse_height,
            extraction_frame_num,
            display_frame_num,
            compute_extraction_btn,
            load_rois_btn,
            draw_rois_btn,
            indicator2,
            show_adv_extraction,
            adv_extraction,
            pn.pane.Markdown('### Save', height=40),
            save_session_btn,
            save_session_and_move_btn,
        )

        # define widget containing plots of arena and extraction
        self.plotting_col = hv.DynamicMap(session_data.display).opts(framewise=True)

        self._layout = pn.Row(self.gui_col, self.plotting_col)

    def __panel__(self):
        return self._layout
