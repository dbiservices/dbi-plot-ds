#!/usr/bin/python3
"""
February 2023, cec@dbi-services.com:
Generic plot script to quickly visualize graphically time-series data;
#See on-line help:
   genplot.py --help
"""
# imports;
from typing import Final
import math
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import mplcursors

# a few constants;
# that many graphs per row;
DEFAULT_FILE_FORMAT: Final = "svg"
DEFAULT_GRAPHS_PER_ROW: Final = 3

# supported graph types:
# buckets graphs plot data sets which have already been bucketized, i.e. they are 2-column lines with the values in X and their frequency in Y;
# instead of histograms, they'll be plotted as bars;
GRAPH_TYPES: Final = ("timeseries", "histogram", "bar", "line", "scatter", "stem", "pie", "buckets")

# file format to save the graphs as;
FORMAT_CHOICES: Final = ['raw', 'rgba', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'ps', 'pgf', 'pdf', 'eps', 'svg', 'svgz']

# dates in data sets must used this supported format;
DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

# send messages to stdout ?
logging = False

def message(text):
   if not logging:
      return
   print(text)

def show_figure(graph, fig, is_windowed = None):
   plt.show(block = False)
   if graph["is-interactive"]:
      if not is_windowed:
         # display the graph full screen;
         pass
      # optimize the elements' position;
      plt.tight_layout()
      # a typical set of values, can be overriden by the caller on return;
      plt.subplots_adjust(hspace = 0.260, wspace = 0.055, top = 0.949, bottom = 0.083, left = 0.026, right = 0.994)
      fig.canvas.flush_events()
      plt.show(block = False)

def saving_logic(save_as, save_as_format):
   global FORMAT_CHOICES
   b_replace = False
   if save_as is not None:
      if "!" == save_as[0:1]:
         save_as = save_as[1:]
         b_replace = True
      else:
         b_replace = False
      if save_as_format is None and save_as is not None:
         # must save graphs but format is missing; let's see if the extension can help;
         extension = None
         if (pos := save_as.rfind(".")) > -1 and (extension := save_as[pos + 1:]) in FORMAT_CHOICES:
            save_as_format = extension
         else:
            # no extension or invalid one given;
            print(f'missing or invalid extension {extension if extension else ""}')
            save_as_format = DEFAULT_FILE_FORMAT
   else:
      save_as_format = DEFAULT_FILE_FORMAT
   return save_as, b_replace, save_as_format

def parse_ok():
   import argparse
   import textwrap
   parser = argparse.ArgumentParser(
      formatter_class = argparse.RawDescriptionHelpFormatter,
      description = textwrap.dedent("""\
Generic plot script to quickly visualize graphically time-series, categorical or single column data;
Also, plot the data's tentative statistical distributions (normal, poisson, exponential, and uniform).
In addition, the distfit package is used for exploring even more theoretical probability distributions.
Usage:
   genplot.py --files=|-f file{,file} {--files=file{,file}}
              [--supertitle=|-t ...]
              [--titles=|-tt title1{,title2}]
              [--save-as=|-s [!]file{,[!]file}]
              [--save-as-format=-sf format{,format}]
              [--save-all-as=|-sa[!]file
              [--save-all-as-format=|-saf format]]
              [--save-no-file|-n]
              [--pie-geometries=|-pg nb_rows X nb_cols]
              [--geometry=|-g nb_rows X nb_cols]
              [--interactive|-i]
              [--crosshair-cursor|-x]
              [--no-stats|-nst]
              [--no-distfit|-ndf]
              [--stats-only|-sto]
              [--distfit-only|-dfo]
              [--log|-l]
where files are text files containing the data with the format below:
1st line: graph_type | title | x-axis label | y-axis label | min-x | max-x | min-y | max-y | legend [| [!]save-as | save-as-format] [ | nb-bins]
e.g.:
   timeseries|Time to Rendition|time|duration [s]|||||legends|!test.svg|svg
graph_type is one of the supported graphs, e.g. timeseries, histogram, bar, line, scatter, stem, pie, buckets; cf. the global GRAPH_TYPES;
title is the individual graph's title;
x- and y- labels are the respective axis' label.
min-x .. max-x define the subset of the data set to plot; min-y .. max-y define a lower and upper clipping zone so the relevant data are scaled visibly.
legend is the data set's legend.
the optional save-as is the file name to save the individual graph into, with save-as-format being one of bitmap formats png, jpeg, tiff and vector formats pdf, eps and svg; cf. the global FORMAT_CHOICES.
if save-as-format is missing, the format will be inferred from save-as' extension.
if save-as is prefixed by an exclamation point, the file is overwritten if it already exists, otherwise a timestamp suffix is added to the file name before the optional extension to make it unique.
if not provided, the graph(s) will not be saved.
nb-bins is used for histograms only and is the number of buckets the data will be binned into, which is also the number of displayed vertical bars.
>= 2nd line: the data formatted as follows (depending on the graph_type).
for timeseries:
  timestamp,value [, anything else on the line is ignored]
e.g.
   14-02-2023 09:51:00,5
   14-02-2023 09:51:02,3
   14-02-2023 09:51:06,2
   ...
timestamp format must be full ISO yyyy-mm-dd hh:mi:ss, i.e. "%Y-%m-%d %H:%M:%S"
for histograms:
this graph shows a statistical distribution; this is not the same as bar graphs, which just plot a vertical bar at x coordinate with height y.
  value [, anything else on the line is ignored]
e.g.
   5
   3
   1
   ...
for bars, lines, scatters, stems, buckets:
x-value,y-value
for pies:
slice-label,value[,e]
The optional "e" stands for "exploded", i.e. if the corresponding slice must be offset from the pie for visibility. 
...
Each --files=... argument groups into one graph and page the data sets from each listed file (minimum 2 files per group).
So, for example, if there are 4 --files=... command-line parameters such as --files=f1,f2 --files=f3 --files=f4,f5,f6 --files=f7,f8, 3 graphs will be produced in its own page.
Then, f1 and f2 will be plotted in the same 1st graph, f3 will be ignored as it is the only file in the group, f4, f5 and f6 in the same 2nd graph, and f7 and f8 in the same 3rd graph.
Titles is an optional list of titles, one per file group, i.e. one for each --files argument.
One restriction is that all the data sets in the same group must request the same graph type, i.e. it is not possible to mix different types of graphs in the same group; use another group for those data sets or just be content with the individual graphs.
--save-as and --save-as-format behave the same way as for individual plots but for each group of graphs and correspond one to one to the respective --files arguments.
To omit a plot's parameter in a group, leave a comma as placeholder.
In addition to individual and grouped graphs, a global page containing all the groups is produced.
--supertitle is the title of that page.
--save-all-as and --save-all-as-format work the same as save-as and save-as-format, but for all the global page.
If --save-no-file is present, no graph files will be written, which is handy when investigating and no report is needed at that stage.
--geometry defined the layout of the graphs in the global page as nb_rows rows x nb_cols columns of graphs; if missing, a default one is used consisting in up to DEFAULT_GRAPHS_PER_ROW (set to 3 presently) and as many lines as needed.
--pie-geometries works identically but for pie charts; since grouped pie charts cannot be stacked, they are each plotted in their own graph in a page with the layout defined in pie-geometries; this is a repeating parameter because there can be several groups of pie charts.
If the optional --interactive argument is present, the graphs will be shown on screen in their own window and the program will pause until the user strikes the Enter key; all the opened graph windows will then be closed and the program exited.
If --interactive is omitted, the program will generate the graphs but not display them; the intention is to save them to disk files if allowed (i.e. --save-no-file is not present and the groups' --save-as are present and/or an output file is present in the data set file(s).
if --crosshair-cursor is present, a crosshair cursor will be enabled that displays the value of the point under the mouse cursor; this is for interactive use only.
As statistical graphs can take some time to be generated for large data sets, it is possible to prevent them with the --no-stats and --no-distfit options.
Conversely, it the individual and group graphs are not necessary, they can be prevented through the --stats-only and --distfit-only options.
It the optional --log is present, informative messages will be sent to stdout along with a dump of the program's data structures used to store the data sets and parameters.
Since thet are named, parameters can be given in any order.
Examples of invocation:
   genplot.py --files=data_1,data_2,data_3 --files=data_2,data_4 --files=data5 --save-as=graph1,,graph2 --supertitle="Performances" --geometry=1x3 save-all_as=!./performances.svg --save-all-as-format=svg
   genplot.py --files=xxs,yys --files=yys,zzs --save-as='!pie1,pie2' --pie-geometries=1x2 --save-as-format=pdf,svg,2 --geometry=1x2 -t the_title --save-all-as='!qqqq' --save-all-as-format=pdf --crosshair-cursor --interactive --no-stats
   genplot.py --files=time_to_render1.csv,time_to_render2.csv --files=time_to_render1_50.csv,time_to_render2_50.csv --titles=graph_group_1,graph_group_2 --save-as='!both1' --save-as-format=pdf,svg,2 --geometry=1x1 -t the_title --save-all-as='!qqqq' --save-all-as-format=pdf --crosshair-cursor --interactive
   """))
   parser.add_argument('--files', '-f',
                       dest = "input_files_group",
                       required = True,
                       nargs = '+',
                       action = 'append',
                       help = 'Specify a file or a comma-separated list of files for the data to plot in a group.')
   parser.add_argument('--supertitle', '-t',
                       dest = "supertitle",
                       required = False,
                       help = 'This is the title of the global page that displays all the grouped graphs.')
   parser.add_argument('--titles', '-tt',
                       dest = "titles",
                       required = False,
                       help = "This is the title of each graph group, i.e. when plotting several data sets in the same graph, each one graph will have a title from that list.")
   parser.add_argument('--save-as', '-s',
                       dest = "save_as",
                       required = False,
                       help = 'This is the name of the output file to optionally save the corresponding graph group into. Prefix it with an exclamation mark ! to replace any existing file with the same name.')
   parser.add_argument('--save-as-format', '-sf',
                       dest = "save_as_format",
                       required = False,
                       help = f'This is the optional format of the file to save the corresponding graph group. It must be one of {FORMAT_CHOICES}. Default format is svg.')
   parser.add_argument('--save-all-as', '-sa',
                       dest = "save_all_as",
                       required = False,
                       help = 'This is the name of the output file to optionally save the global page of group graphs into. Prefix it with an exclamation mark ! to replace any existing file with the same name.')
   parser.add_argument('--save-all-as-format', '-saf',
                       dest = "save_all_as_format",
                       required = False,
                       choices = FORMAT_CHOICES,
                       help = f'This is the optional format of the file to save the global page of graphs into. It must be one of {FORMAT_CHOICES}. Default format is svg.')
   parser.add_argument('--save-no-file', '-n',
                       dest = "save_no_file",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, no graphs are saved to disk files.')
   parser.add_argument('--pie-geometries', '-pg',
                       dest = "pie_geometries",
                       required = False,
                       help = 'If several pie charts in a group are to be displayed on the same page, specify how many across the page and how many vertically. Syntax: nb_rows [xX] nb_cols; e.g. 3x2.\nDefault nx3, such as 3n >= nb graphs in the group, empty graphs possible if 3n > nb graphs.')
   parser.add_argument('--geometry', '-g',
                       dest = "geometry",
                       required = False,
                       help = 'If several goup graphs are to be displayed on the global page, specify how many across the page and how many vertically. Syntax: nb_rows [xX] nb_cols; e.g. 3x2.\nDefault nx3.')
   parser.add_argument('--interactive', '-i',
                       dest = "is_interactive",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, each graph (including its plot(s) will be opened in its own interactive window.')
   parser.add_argument('--crosshair-cursor', '-x',
                       dest = "crosshair_cursor",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, a crosshair cursor is shown that displays the value under the cursor. For interactive work.')
   parser.add_argument('--no-stats', '-nst',
                       dest = "no_stats",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, stats graphs are not produced.')
   parser.add_argument('--no-distfit', '-ndf',
                       dest = "no_distfit",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, distfit graphs are not produced.')
   parser.add_argument('--stats-only', '-sto',
                       dest = "stats_only",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, only stats graphs are produced, i.e. no individual or group graphs. This parameter has more priority that --no-stats. Can coexist with --distfit-only.')
   parser.add_argument('--distfit-only', '-dfo',
                       dest = "distfit_only",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, only distfit graphs are produced, i.e. no individual or group graphs. This parameter has more priority that --no-distfit. Can coexist with --stats-only.')
   parser.add_argument('--log', '-l',
                       dest = "logging",
                       required = False,
                       action = 'store_true',
                       help = 'Optional. If present, messages are sent to stdout.')
   args = parser.parse_args()

   def check_geometry(nb_graphs, geometry = None):
      import re
      if geometry is None:
         # let's first fill a row of DEFAULT_GRAPHS_PER_ROW graphs, and then spill the rest over the next lines;
         nb_cols = nb_graphs if nb_graphs <= DEFAULT_GRAPHS_PER_ROW else DEFAULT_GRAPHS_PER_ROW
         nb_rows = (nb_graphs - 1) // nb_cols + 1
      else:
         re_geom = re.compile("^(\d+) *[xX] *(\d+)$")
         parse_geom = re_geom.findall(geometry)
         if not parse_geom:
            print(f"illegal geometry {geometry}, exiting ...")
            return None
         nb_rows, nb_cols = parse_geom[0]
      return int(nb_rows), int(nb_cols)

   # check each group's geometry (used by pie charts only);
   pie_geometries = []
   if args.pie_geometries:
      geometries = args.pie_geometries.split(",")
      for g in geometries:
         nb_rows, nb_cols = check_geometry(geometry = g, nb_graphs = len(geometries))
         if nb_rows:
            pie_geometries.append([nb_rows, nb_cols])
         else:
            return None
   else:
      [pie_geometries.append(check_geometry(nb_graphs = len(_[0].split(",")))) for _ in args.input_files_group]

   # check geometry of whole page graph;
   nb_graphs = len(args.input_files_group)
   nb_rows, nb_cols = check_geometry(geometry = args.geometry, nb_graphs = nb_graphs)
   if not nb_rows:
      return None
   save_all_as, b_all_replace, save_all_as_format = saving_logic(args.save_all_as, args.save_all_as_format)

   message(f"\nThe command-line arguments were:\n{nb_graphs = }, {args.supertitle = }, {args.input_files_group = }, {args.titles = }, {pie_geometries = }, {args.save_as = }, {args.save_as_format = }, {nb_rows = }, {nb_cols = }, {save_all_as = }, {b_all_replace = }, {save_all_as_format = }, {args.crosshair_cursor = }, {args.is_interactive = }, {args.logging = }, {args.save_no_file = }, {args.no_stats}, {args.no_distfit}, {args.stats_only}, {args.distfit_only}")
   return args.supertitle, args.input_files_group, args.titles, pie_geometries, args.save_as, args.save_as_format, nb_rows, nb_cols, save_all_as, b_all_replace, save_all_as_format, args.crosshair_cursor, args.is_interactive, args.logging, args.save_no_file, args.no_stats, args.no_distfit, args.stats_only, args.distfit_only

# displays the graph data structure for debugging or info;
# either the global arguments, or a particular group's ones or a particular dataset's ones can be selected;
def show_params(graph = None, which_ones = ("global", "groups", "dataset"), data = None):
   if not logging:
      return

   if "raw" in which_ones:
      print(f'{graph = }')

   if "global" in which_ones:
      print('* processed command-line arguments')
      print(f'{graph["supertitle"] = }')
      print(f'{graph["input-files-group"] = }')
      print(f'{graph["titles"] = }')
      print(f'{graph["pie-geometries"] = }')
      print(f'{graph["save-as-list"] = }')
      print(f'{graph["save-as-format-list"] = }')
      print(f'{graph["nb-rows"] = } x {graph["nb-cols"] = }')
      print(f'{graph["save-as"] = }')
      print(f'{graph["b-replace"] = }')
      print(f'{graph["save-as-format"] = }')
      print(f'{graph["crosshair-cursor"] = }')
      print(f'{graph["is-interactive"] = }')
      print(f'{graph["save-no-file"] = }')
      print(f'{graph["no-stats"] = }')
      print(f'{graph["no-distfit"] = }')
      print(f'{graph["stats-only"] = }')
      print(f'{graph["distfit-only"] = }')
      print('')

   if "groups" in which_ones:
      print("** groups' parameters")
      print(f'{graph["save-as"] = }')
      print(f'{graph["save-as-format"] = }')
      print(f'{graph["b-replace"] = }')
      print('')

   if "dataset" in which_ones:
      print("*** dataset's parameters")
      print(f'{data["file-name"] = }')
      print(f'{data["title"] = }')
      print(f"{data = }")
      print(f'{data["all-data"] =}')
      print(f'{data["graph-type"] = }')
      print(f'{data["is-categorical-in-x"] = } (unconfirmed yet)')
      print(f'{data["is-categorical-in-y"] = } (unconfirmed yet)')
      print(f'{data["x-axis-label"] = }')
      print(f'{data["y-axis-label"] = }')
      print(f'{data["min-x"] = }')
      print(f'{data["max-x"] = }')
      print(f'{data["min-y"] = }')
      print(f'{data["max-y"] = }')
      print(f'{data["legends"] = }')
      print(f'{data["save-as"] = }')
      print(f'{data["b-replace"] = }')
      print(f'{data["save-as-format"] = }')
      if "nb-bins" in data:
         print(f'{data["nb-bins"] = }')
      if "exploded" in data:
         print(f'{data["exploded"] = }')
      print(f'{data["all-data"] = }')
      print('')

def save_graph(file_name, file_format, graph):
   if graph["save-no-file"]:
      return
   save_as = file_name
   if save_as is not None:
      import os
      if os.path.exists(save_as) and not graph["b-replace"]:
         # the output file already exist and should not be overwritten: append a timstamp suffix to the file name;
         if (pos := save_as.rfind(".")) > -1:
            save_as = f'{save_as[:pos]}_{datetime.now().strftime("%Y%m%d%H%M%S")}.{save_as[pos + 1: ]}'
         else:
            save_as = f'{save_as}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
      # else: output file does not exist yet, or it can be overwritten;
      plt.savefig(fname = f'{save_as}', format = file_format, bbox_inches = "tight", facecolor = 'auto', edgecolor = 'auto')

# ----------------------------------------------------------------------------------------------------------------------
# timeseries stuff;
def do_plot_individual_timeseries(plot, axis):
   global graph
   axis.set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axis.set_ylabel(plot["y-axis-label"], fontsize = "small", fontweight = "bold")
   axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d")) # %H:%M:%S"))
   for label in axis.get_xticklabels(which = 'major'):
      label.set(rotation = 45, horizontalalignment = 'right', fontsize='small')
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis.plot_date(plot["x"], plot["y"], label = plot["legends"]), hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, plot = plot):
         # if several graphs are plotted at once, we need to preserve the context of the cursor in the callback, i.e. on which graph it is applied;
         # hence the extra parameter plot that receives the specific plot at definition time and uses it when invoked as a callback;
         sel.annotation.set(text = f'({plot["x"][sel.index]}, {plot["y"][sel.index]})')
   else:
      axis.plot_date(plot["x"], plot["y"], label = plot["legends"])

def do_plot_group_timeseries(gr, graph_index, axis, show_graph_title = True):
   global graph
   # in group graphs, only the supertitle makes sense;
   if show_graph_title and graph["titles"] and graph_index < len(graph["titles"]):
      axis.set_title(graph["titles"][graph_index], fontsize = "large", fontweight = "bold")
   has_legends = False
   window_title = "files "
   for counter, plot in enumerate(gr["plot"]):
      if plot["legends"]:
         has_legends = True
      do_plot_individual_timeseries(plot, axis)
      window_title = f'{window_title}{" | " if counter > 0 else ""}{plot["title"]}'
   if has_legends:
      axis.legend()
   return window_title

def plot_individual_timeseries(plot):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   plt.get_current_fig_manager().set_window_title(f'{plot["title"]} - file {plot["file-name"]}')
   axis = axs.flat[0]
   axis.set_title(plot["title"], fontsize = "large", fontweight = "bold")
   do_plot_individual_timeseries(plot, axis)
   if plot["legends"]:
      axis.legend()
   save_graph(plot["save-as"], plot["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# plots the given group into its own figure;
def plot_group_timeseries(gr, graph_index):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   axis = axs.flat[0]
   window_title = do_plot_group_timeseries(gr, graph_index, axis, show_graph_title = False)
   if graph["titles"] and graph_index < len(graph["titles"]):
      plt.get_current_fig_manager().set_window_title(f'{graph["titles"][graph_index]} - {window_title}')
      fig.suptitle(graph["titles"][graph_index], fontsize = "x-large", fontweight = "bold")
   else:
      plt.get_current_fig_manager().set_window_title(f'{window_title}')
   save_graph(gr["save-as"], gr["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# ----------------------------------------------------------------------------------------------------------------------
# histograms stuff;
# plot a single or several dataset(s) as histogram(s) into the graph identified by axis;
def do_plot_individual_histogram(graph, plot, axis, nb_bins = None, nb_datasets = 1):
   # return the minimum number of buckets to use for data set X;
   # if uniq_values_with_original_order is not empty (i.e. when plotting multiple data sets), the buckets are appended to it;
   # the end result is as if the data sets' files where concatened;
   from collections import Counter
   def find_min_nb_bins(X, uniq_values_with_original_order):
      uniq_values = {v:1 for v in uniq_values_with_original_order}
      # like collections.Counter() but preserves the original order of appearance in the data set's input file;
      # that's the order used by matplotlib;
      for x in list(Counter(X)):
         if x in uniq_values:
            uniq_values.update({x:uniq_values[x] + 1})
         else:
            uniq_values.update({x:1})
            uniq_values_with_original_order.append(x)
      return uniq_values_with_original_order

   if 1 == nb_datasets:
      axis.set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
      axis.set_ylabel(plot["y-axis-label"], fontsize = "small", fontweight = "bold")
      tot = len(plot["x"])
      legends = plot["legends"]
      X = plot["x"]
      is_categorical_in_x = plot["is-categorical-in-x"]
      max_bins = len(Counter(X))
      uniq_values_with_original_order = []
      uniq_values_with_original_order = find_min_nb_bins(X, uniq_values_with_original_order)
      print(f'1: {uniq_values_with_original_order=}')
   else:
      # if several data sets are to be plot (i.e. from do_plot_group_histograms), take this info from the first one;
      axis.set_xlabel(plot[0]["x-axis-label"], fontsize = "small", fontweight = "bold")
      axis.set_ylabel(plot[0]["y-axis-label"], fontsize = "small", fontweight = "bold")
      tot = []
      legends = []
      X = []
      is_categorical_in_x = False
      max_bins = 0
      uniq_values_with_original_order = []
      for i, p in enumerate(plot):
         tot.append(len(p["x"]))
         legends.append(p["legends"])
         X.append([x for x in p["x"]])
         is_categorical_in_x = is_categorical_in_x or p["is-categorical-in-x"]
         max_bins = max_bins if max_bins > (l := len(Counter(p["x"]))) else l
         uniq_values_with_original_order = find_min_nb_bins(p["x"], uniq_values_with_original_order)
         print(f'multiple plots, {i}: {uniq_values_with_original_order=}')
   for label in axis.get_xticklabels(which = 'major'):
      label.set(rotation = 90, horizontalalignment = 'right', fontsize='small')
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      # crosshair cursor in hist graphs is more complex;
      # while the y positions returned by hist() are correct, the returned x positions are approximate;
      # therefore, we display an unobtrusive scatter plot over the bars and apply the crosshair cursor to it;
      # if multiple data sets, all the histograms are plotted at once;
      values = axis.hist(X, histtype = 'bar', density = False, bins = sorted([x for x in uniq_values_with_original_order]) if not is_categorical_in_x else list(range(len(uniq_values_with_original_order) + 1)), linewidth = 0.5, edgecolor = "white", label = legends, align = 'left')
      axis.autoscale(tight = True)
      x_values = values[1]
      y_values = values[0]

      def compute_tit_pos(dataset = 0):
         tmp = x_values[:]
         new_x_values = []
         for j, v in enumerate(tmp[:-1]):
            bar_width = (tmp[j + 1] - v) / (nb_datasets + 1)
            #x = v + (dataset + 1) * bar_width
            x = v + (dataset - 1) * bar_width
            new_x_values.append(x)
         return new_x_values

      if not isinstance(y_values[0], float):
         # or if nb_datasets > 1 since nb_datasets is equal to dim(y);
         # multi-plot graph, y_values is an array of arrays of floats;
         for dataset, y in enumerate(y_values):
            new_x_values = compute_tit_pos(dataset)
            cursor = mplcursors.cursor(axis.scatter(new_x_values, y, s = 1), hover = mplcursors.HoverMode.Transient)
            @cursor.connect("add")
            def on_add(sel, tot = tot[dataset], x_values = x_values[:], y_values = y[:]):
               index = int(sel.index)
               sel.annotation.set(text = f'({x_values[index] if not is_categorical_in_x else uniq_values_with_original_order[index]}, {y_values[index]:.2f}={100*y_values[index]/tot:.2f}%)')
      else:
         # one-plot graph, y_values is an array of floats;
         new_x_values = compute_tit_pos()
         cursor = mplcursors.cursor(axis.scatter(new_x_values, y_values, s = 1), hover = mplcursors.HoverMode.Transient)
         @cursor.connect("add")
         def on_add(sel, tot = tot, x_values = new_x_values[:], y_values = y_values[:]):
            index = int(sel.index)
            sel.annotation.set(text = f'({x_values[index] if not is_categorical_in_x else uniq_values_with_original_order[index]}, {y_values[index]:.2f}={100*y_values[index]/tot:.2f}%)')
            # add the percentages in the y-axis but only when a single data set is plotted in order to leave as much space as possible to multiset plots;
            yticks = axis.get_yticks()
            yticks = [f'{y}={100*(y/tot):.2f}%' for y in yticks]
            axis.set_yticklabels(yticks)
      return values
   else:
      values = axis.hist(X, bins = max_bins if not plot["is-categorical-in-x"] else list(range(len(uniq_values_with_original_order))), histtype = 'bar', density = False, linewidth = 0.5, edgecolor = "white", label = label)
      if 1 == nb_datasets:
         # only meaningfull for one ds, unless the others have the same number of points;
         yticks = axis.get_yticks()
         yticks = [f'{y}={100*(y/tot):.2f}%' for y in yticks]
         axis.set_yticklabels(yticks)
      return values

# plot the given datasets as a histogram into a new figure;
def plot_individual_histogram(plot):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   plt.get_current_fig_manager().set_window_title(f'{plot["title"]} - file {plot["file-name"]}')
   axis = axs.flat[0]
   axis.set_title(plot["title"], fontsize = "large", fontweight = "bold")
   # avoid too many unnecessary bins;
   nb_bins = min(len(plot["x"]), plot["nb-bins"] if "nb-bins" in plot and plot["nb-bins"] > 0 else len(plot["x"]))
   do_plot_individual_histogram(graph, plot, axis, nb_bins = nb_bins)
   if plot["legends"]:
      axis.legend()
   save_graph(plot["save-as"], plot["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# plot all the given group's datasets as a histogram into the given axis;
def do_plot_group_histograms(gr, graph_index, axis, show_graph_title = True):
   global graph
   # in group graphs, only the supertitle makes sense;
   if show_graph_title and graph["titles"] and graph_index < len(graph["titles"]):
      axis.set_title(graph["titles"][graph_index], fontsize = "large", fontweight = "bold")
   legends = []
   # avoid too many unnecessary bins;
   # if plot["nb_bins"] not defined or set to 0, use the data set's size;
   nb_bins = max([min(len(plot["x"]), plot["nb-bins"] if "nb-bins" in plot and plot["nb-bins"] > 0 else len(plot["x"])) for plot in gr["plot"]])
   window_title = "files "
   plot_group = []
   for counter, plot in enumerate(gr["plot"]):
      if plot["legends"]:
         legends.append(plot["legends"])
      plot_group.append(plot)
      window_title = f'{window_title}{" | " if counter > 0 else ""}{plot["title"]}'
   do_plot_individual_histogram(graph, plot_group, axis, nb_bins = nb_bins, nb_datasets = counter + 1)
   if legends:
      axis.legend()
      pass
   return window_title

# plot the given group's dataset into a histogram in its own figure;
def plot_group_histograms(gr, graph_index):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   axis = axs.flat[0]
   window_title = do_plot_group_histograms(gr, graph_index, axis, show_graph_title = False)
   if graph["titles"] and graph_index < len(graph["titles"]):
      plt.get_current_fig_manager().set_window_title(f'{graph["titles"][graph_index]} - {window_title}')
      fig.suptitle(graph["titles"][graph_index], fontsize = "x-large", fontweight = "bold")
   else:
      plt.get_current_fig_manager().set_window_title(f'{window_title}')
   save_graph(gr["save-as"], gr["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# ----------------------------------------------------------------------------------------------------------------------
# scatter stuff;
# plot a single dataset as a scatter graph into the given axis;
def do_plot_individual_scatter(plot, axis):
   global graph
   axis.set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axis.set_ylabel(plot["y-axis-label"], fontsize = "small", fontweight = "bold")
   for label in axis.get_xticklabels(which = 'major'):
      label.set(rotation = 45, horizontalalignment = 'right', fontsize='small')
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis.scatter(plot["x"], plot["y"], label = plot["legends"]), hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, plot = plot):
         # if several graphs are plotted at once, we need to preserve the context of the cursor in the callback, i.e. on which graph it is applied;
         # hence the extra parameter plot that receives the specific plot at definition time and uses it when invoked as a callback;
         # we take advantage of the fact that default parameters are evaluated at definition time in python;
         sel.annotation.set(text = f'({plot["x"][sel.index]}, {plot["y"][sel.index]})')
   else:
      axis.scatter(plot["x"], plot["y"], label = plot["legends"])

# plot all the given group's datasets as a scatter graph into the given axis;
def do_plot_group_scatters(gr, graph_index, axis, show_graph_title = True):
   global graph
   # in group graphs, only the supertitle makes sense;
   if  show_graph_title and graph["titles"] and graph_index < len(graph["titles"]):
      axis.set_title(graph["titles"][graph_index], fontsize = "large", fontweight = "bold")
   has_legends = False
   window_title = "files "
   for counter, plot in enumerate(gr["plot"]):
      if plot["legends"]:
         has_legends = True
      do_plot_individual_scatter(plot, axis)
      window_title = f'{window_title}{" | " if counter > 0 else ""}{plot["title"]}'
   if has_legends:
      axis.legend()
   return window_title

# plot the given dataset as a scatter graph into a new figure;
def plot_individual_scatter(plot):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   #plt.get_current_fig_manager().set_window_title(plot["title"])
   plt.get_current_fig_manager().set_window_title(f'{plot["title"]} - file {plot["file-name"]}')
   axis = axs.flat[0]
   axis.set_title(plot["title"], fontsize = "large", fontweight = "bold")
   do_plot_individual_scatter(plot, axis)
   if plot["legends"]:
      axis.legend()
   save_graph(plot["save-as"], plot["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# plot the given group's dataset into a scatter graph in its own figure;
def plot_group_scatters(gr, graph_index):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   axis = axs.flat[0]
   window_title = do_plot_group_scatters(gr, graph_index, axis, show_graph_title = False)
   if graph["titles"] and graph_index < len(graph["titles"]):
      plt.get_current_fig_manager().set_window_title(f'{graph["titles"][graph_index]} - {window_title}')
      fig.suptitle(graph["titles"][graph_index], fontsize = "x-large", fontweight = "bold")
   else:
      plt.get_current_fig_manager().set_window_title(f'{window_title}')
   save_graph(gr["save-as"], gr["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# ----------------------------------------------------------------------------------------------------------------------
# stem (aka lollipop) stuff;
# plot a single dataset as a stem graph into the given axis;
def do_plot_individual_stem(plot, axis, line_color = None):
   axis.set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axis.set_ylabel(plot["y-axis-label"], fontsize = "small", fontweight = "bold")
   axis.set_ylim(top = plot["min-y"])
   for label in axis.get_xticklabels(which = 'major'):
      label.set(rotation = 45, horizontalalignment = 'right', fontsize='small')
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      if line_color:
         cursor = mplcursors.cursor(stem_container := axis.stem(plot["x"], plot["y"], label = plot["legends"], linefmt = line_color, bottom = plot["min-y"]), hover = mplcursors.HoverMode.Transient)
         plt.setp(stem_container.markerline, 'color', line_color)
      else:
         cursor = mplcursors.cursor(axis.stem(plot["x"], plot["y"], label = plot["legends"], bottom = plot["min-y"]),  hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, plot = plot):
         # if several graphs are plotted at once, we need to preserve the context of the cursor in the callback, i.e. on which graph it is applied;
         # hence the extra parameter plot that receives the specific plot at definition time and uses it when invoked as a callback;
         try:
            # get rid of puzzling "TypeError: list indices must be integers or slices, not tuple" error;
            sel.annotation.set(text = f'({plot["x"][sel.index]}, {plot["y"][sel.index]})')
         except:
            None
   else:
      stem_container = axis.stem(plot["x"], plot["y"], label = plot["legends"], linefmt = line_color, bottom = plot["min-y"])
      if line_color:
         plt.setp(stem_container.markerline, 'color', line_color)

# plot all the given group's datasets as a stem graph into the given axis;
def do_plot_group_stems(gr, graph_index, axis, show_graph_title = True):
   global graph
   import matplotlib.colors as mcolors
   from cycler import cycler
   CSS4_color_names = list(mcolors.CSS4_COLORS)
   CSS4_color_cycler = cycler(color = CSS4_color_names)
   # in group graphs, only the supertitle makes sense;
   if show_graph_title and graph["titles"] and graph_index < len(graph["titles"]):
      axis.set_title(graph["titles"][graph_index], fontsize = "large", fontweight = "bold")
   axis.set_prop_cycle(CSS4_color_cycler)
   has_legends = False
   window_title = "files "
   # start the color cycler with not too light a color;
   color_index = 10
   for counter, plot in enumerate(gr["plot"]):
      if plot["legends"]:
         has_legends = True
      do_plot_individual_stem(plot, axis, line_color = CSS4_color_names[color_index])
      window_title = f'{window_title}{" | " if counter > 0 else ""}{plot["title"]}'
      color_index += 1
   if has_legends:
      axis.legend()
   return window_title

# plot the given datasets as a stem graph into a new figure;
def plot_individual_stem(plot):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   plt.get_current_fig_manager().set_window_title(f'{plot["title"]} - file {plot["file-name"]}')
   axis = axs.flat[0]
   axis.set_title(plot["title"], fontsize = "large", fontweight = "bold")
   do_plot_individual_stem(plot, axis)
   if plot["legends"]:
      axis.legend()
   save_graph(plot["save-as"], plot["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# plot the given group's dataset into a stem graph in its own figure;
def plot_group_stems(gr, graph_index):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   axis = axs.flat[0]
   window_title = do_plot_group_stems(gr, graph_index, axis, show_graph_title = False)
   if graph["titles"] and graph_index < len(graph["titles"]):
      plt.get_current_fig_manager().set_window_title(f'{graph["titles"][graph_index]} - {window_title}')
      fig.suptitle(graph["titles"][graph_index], fontsize = "x-large", fontweight = "bold")
   else:
      plt.get_current_fig_manager().set_window_title(f'{window_title}')
   save_graph(gr["save-as"], gr["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# ----------------------------------------------------------------------------------------------------------------------
# line stuff;
# plot a single dataset as a line graph into the given axis;
def do_plot_individual_line(plot, axis):
   global graph
   axis.set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axis.set_ylabel(plot["y-axis-label"], fontsize = "small", fontweight = "bold")
   for label in axis.get_xticklabels(which = 'major'):
      label.set(rotation = 45, horizontalalignment = 'right', fontsize = 'small')
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis.plot(plot["x"], plot["y"], label = plot["legends"]), hover = mplcursors.HoverMode.Transient) #, highlight = True) # too distracting ...
      @cursor.connect("add")
      def on_add(sel, plot = plot):
         # if several graphs are plotted at once, we need to preserve the context of the cursor in the callback, i.e. on which graph it is applied;
         # hence the extra parameter plot that receives the specific plot at definition time and uses it when invoked as a callback;
         indx = int(sel.index)
         sel.annotation.set(text = f'({plot["x"][indx]}, {plot["y"][indx]})')
   else:
      axis.plot(plot["x"], plot["y"], label = plot["legends"])

# plot all the given group's datasets as a line graph into the given axis;
def do_plot_group_lines(gr, graph_index, axis, show_graph_title = True):
   global graph
   # in group graphs, only the supertitle makes sense;
   if show_graph_title and graph["titles"] and graph_index < len(graph["titles"]):
      axis.set_title(graph["titles"][graph_index], fontsize = "large", fontweight = "bold")
   has_legends = False
   window_title = "files "
   for counter, plot in enumerate(gr["plot"]):
      if plot["legends"]:
         has_legends = True
      do_plot_individual_line(plot, axis)
      window_title = f'{window_title}{" | " if counter > 0 else ""}{plot["title"]}'
   if has_legends:
      axis.legend()
   return window_title

# plot the given dataset as a line graph into a new figure;
def plot_individual_line(plot):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   plt.get_current_fig_manager().set_window_title(f'{plot["title"]} - file {plot["file-name"]}')
   axis = axs.flat[0]
   axis.set_title(plot["title"], fontsize = "large", fontweight = "bold")
   do_plot_individual_line(plot, axis)
   if plot["legends"]:
      axis.legend()
   save_graph(plot["save-as"], plot["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# plot the given group's dataset into a line graph in its own figure;
def plot_group_lines(gr, graph_index):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   axis = axs.flat[0]
   window_title = do_plot_group_lines(gr, graph_index, axis, show_graph_title = False)
   if graph["titles"] and graph_index < len(graph["titles"]):
      plt.get_current_fig_manager().set_window_title(f'{graph["titles"][graph_index]} - {window_title}')
      fig.suptitle(graph["titles"][graph_index], fontsize = "x-large", fontweight = "bold")
   else:
      plt.get_current_fig_manager().set_window_title(f'{window_title}')
   save_graph(gr["save-as"], gr["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# ----------------------------------------------------------------------------------------------------------------------
# bar stuff;
BAR_WIDTH: Final = 0.25

# plot a single dataset as a bar graph into the given axis;
# unlike the other do_plot_* functions, this one takes graph as parameter because the function is also invoked from within a sub-process and unlike threads, detached sub-processes don't see global variables and cannot access them;
# x_data and y_data are both None when the caller is plot_individual_bar() and not None when the caller is do_plot_group_bars();
# offset is used for multiple data sets; each bar is plotted with the given offset to the right; offset defaults to 0;
#def do_plot_individual_bar(graph, plot, axis, x_data = None, y_data = None, offset = None, bottom = None):
def do_plot_individual_bar(graph, plot, axis, x_data = None, y_data = None, offset = 0, bottom = None, xticks_labels = None):
   axis.set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axis.set_ylabel(plot["y-axis-label"], fontsize = "small", fontweight = "bold")
   for label in axis.get_xticklabels(which = 'major'):
      label.set(rotation = 45, horizontalalignment = 'right', fontsize='small')
   if "buckets" == plot["graph-type"] and not plot["is-categorical-in-y"]:
      # the data set has been pre-buckerized; thus, percentages label make sense here;
      tot = sum(plot["y"])
   else:
      tot = None
   axis.grid(visible = True, color = '0', linestyle = '--', linewidth = 1, which = 'both', axis = 'y', alpha = 0.5)
   # let's use x_data for plot["x"] or all data sets' merged plot["x"] (the latter ones are natural integer place-holders);
   if x_data is None:
      # called by plot_individual_bar() for one data set;
      x_data = plot["x"]
      y_data = plot["y"]
      # bug in X-categorical plots: the Y-values are incremented by the bottom value when displayed;
      # the work-around is to keep bottom (otherwise, it'll defaul to 0) but decrease the Y-values by the bottom amount;
      # this work-around is compatible with non-categorical data so we apply it indifferently;
      # bottom must be applied to all the data sets in same graph;
      #wa_Y = [y - (bottom if bottom is not None else plot["min-y"]) + 1 for y in plot["y"]]
      wa_Y = y_data
   else:
      # called by do_plot_group_bars();
      # bottom is always set;
      #wa_Y = [y - bottom + 1 for y in y_data]
      wa_Y = y_data
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis.bar([x + offset for x in x_data] if not plot["is-categorical-in-x"] else [i + offset for i, _ in enumerate(x_data)], wa_Y, width = BAR_WIDTH, label = plot["legends"]), hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, plot_x = x_data if not xticks_labels else xticks_labels, plot_y = y_data, tot = tot):
         # if several graphs are plotted at once, we need to preserve the context of the cursor in the callback, i.e. on which graph it is applied;
         # hence the extra parameters plot_x and plot_y that receive the specific point's coordinates at definition time and uses it when invoked as a callback;
         indx = int(sel.index)
         if "buckets" == plot["graph-type"] and not plot["is-categorical-in-y"]:
            sel.annotation.set(text = f'({plot_x[indx]}, {plot_y[indx]}={100*wa_Y[indx]/tot:.2f}%)')
         else:
            sel.annotation.set(text = f'({plot_x[indx]}, {plot_y[indx]})')
   else:
      axis.bar([x + offset for x in x_data] if not plot["is-categorical-in-x"] else [i + offset for i, _ in enumerate(x_data)], wa_Y, width = BAR_WIDTH, label = plot["legends"])#, bottom = bottom)
   if xticks_labels:
      axis.set_xticks([i for i, _ in enumerate(xticks_labels)], xticks_labels)
   elif plot["is-categorical-in-x"]:
      axis.set_xticks([i + int(offset) for i, _ in enumerate(x_data)], x_data)
   #else leave it to matplotlib to place the x-ticks;
   if "buckets" == plot["graph-type"] and not plot["is-categorical-in-y"]:
      # the data set has been pre-buckerized; thus, percentages label make sense here;
      yticks = axis.get_yticks()
      yticks = [f'{y}={100*(y/tot):.2f}%' for y in yticks]
      axis.set_yticklabels(yticks)
   axis.autoscale(tight = True)

# plot all the given group's datasets as a bar graph into the given axis;
def do_plot_group_bars(gr, graph_index, axis, show_graph_title = True):
   global graph
   # in group graphs, only the supertitle makes sense;
   if show_graph_title and graph["titles"] and graph_index < len(graph["titles"]):
      axis.set_title(graph["titles"][graph_index], fontsize = "large", fontweight = "bold")
   has_legends = False
   window_title = "files "
   minY = min([plot["min-y"] for plot in gr["plot"]])
   nb_datasets = len(gr["plot"])
   xticks_labels = []
   merged_x = []
   for plot_index, plot in enumerate(gr["plot"]):
      if plot["is-categorical-in-x"] and nb_datasets > 1:
         for x in plot["x"]:
            if x not in xticks_labels:
               merged_x.append(x)
               xticks_labels.append(x)
   # nicer presentation with sorted x-labels;
   merged_x.sort()
   xticks_labels.sort()
   x_data = [i for i, _ in enumerate(merged_x)]
   null_value_found = False
   for plot_index, plot in enumerate(gr["plot"]):
      if plot["legends"]:
         has_legends = True
      if plot["is-categorical-in-x"] and nb_datasets > 1:
         if plot["is-categorical-in-y"]:
            y_data = [plot["y"][pos] if x in plot["x"] and (pos := plot["x"].index(x)) > -1 else "" for x in merged_x] 
            # little hack so the first empty Y values apepars first in y_data, for nicer presentation;
            if not null_value_found and "" in y_data and (pos := y_data.index("") > -1):
               y_data.remove("")
               y_data.insert(0, "")
               null_value_found = True
         else:
            # fill in the y-axis with missing values set to nan;
            y_data = [plot["y"][pos] if x in plot["x"] and (pos := plot["x"].index(x)) > -1 else np.nan for x in merged_x] 
      #elif plot["is-categorical-in-y"]:
      #   ...
      else:
         x_data = plot["x"]
         y_data = plot["y"]
      do_plot_individual_bar(graph, plot, axis, x_data, y_data, offset = BAR_WIDTH * plot_index, bottom = minY, xticks_labels = xticks_labels)
      window_title = f'{window_title}{" | " if plot_index > 0 else ""}{plot["file-name"]}'
   if has_legends:
      axis.legend()
   return window_title

# plot the given dataset as a bar graph into a new figure;
def plot_individual_bar(plot):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   #plt.get_current_fig_manager().set_window_title(plot["title"])
   plt.get_current_fig_manager().set_window_title(f'{plot["title"]} - file {plot["file-name"]}')
   axis = axs.flat[0]
   axis.set_title(plot["title"], fontsize = "large", fontweight = "bold")
   do_plot_individual_bar(graph, plot, axis)
   if plot["legends"]:
      axis.legend()
   save_graph(plot["save-as"], plot["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# plot the given group's dataset into a bar graph in its own figure;
def plot_group_bars(gr, graph_index):
   global graph
   # generate and save each group graph in its own file;
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   axis = axs.flat[0]
   window_title = do_plot_group_bars(gr, graph_index, axis, show_graph_title = False)
   if graph["titles"] and graph_index < len(graph["titles"]):
      plt.get_current_fig_manager().set_window_title(f'{graph["titles"][graph_index]} - {window_title}')
      fig.suptitle(graph["titles"][graph_index], fontsize = "x-large", fontweight = "bold")
   else:
      plt.get_current_fig_manager().set_window_title(f'{window_title}')
   save_graph(gr["save-as"], gr["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# ----------------------------------------------------------------------------------------------------------------------
# pie stuff;

# plot a single dataset as a pie graph into the given axis;
def do_plot_individual_pie(plot, axis):
   global graph
   axis.set_title(plot["title"], fontsize = "large", fontweight = "bold")
   axis.pie(plot["y"], explode = plot["exploded"], labels = plot["x"], autopct = '%1.1f%%', pctdistance = 0.8)

# plot all the given group's datasets as a pie graph into the given axis;
# as this function is only used from plot_group_pies, it could be moved into it;
def do_plot_group_pies(gr, axs):
   global graph
   window_title = "files "
   for plot_index, plot in enumerate(gr["plot"]):
      axis = axs.flat[plot_index]
      do_plot_individual_pie(plot, axis)
      window_title = f'{window_title}{" | " if plot_index > 0 else ""}{plot["title"]}'
   return window_title

# plot the given dataset as a pie graph into a new figure;
def plot_individual_pie(plot):
   global graph
   fig, axs = plt.subplots(ncols = 1, nrows = 1, squeeze = False)
   #plt.get_current_fig_manager().set_window_title(plot["title"])
   plt.get_current_fig_manager().set_window_title(f'{plot["title"]} - file {plot["file-name"]}')
   axis = axs.flat[0]
   do_plot_individual_pie(plot, axis)
   save_graph(plot["save-as"], plot["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# plot the given group's dataset into a pie graph in its own figure;
def plot_group_pies(gr, graph_index, plot_index):
   global graph
   fig, axs = plt.subplots(nrows = graph["pie-geometries"][plot_index][0], ncols = graph["pie-geometries"][plot_index][1], squeeze = False, figsize = (16, 9))
   window_title = do_plot_group_pies(gr, axs)
   if graph["titles"] and graph_index < len(graph["titles"]):
      plt.get_current_fig_manager().set_window_title(f'{graph["titles"][graph_index]} - {window_title}')
      fig.suptitle(graph["titles"][graph_index], fontsize = "x-large", fontweight = "bold")
   else:
      plt.get_current_fig_manager().set_window_title(f'{window_title}')
   # remove unused axes when geometry is too large;
   for _ in range(len(gr["plot"]), graph["pie-geometries"][plot_index][0] * graph["pie-geometries"][plot_index][1]):
      axis = axs.flat[_]
      axis.remove()
   save_graph(gr["save-as"], gr["save-as-format"], graph)
   axis.autoscale(tight = True)
   show_figure(graph, fig, True)

# ----------------------------------------------------------------------------------------------------------------------
# stacked bar stuff;
def plot_individual_stacked_bar(plot):
   global graph
   None
def  plot_group_stacked_bar(gr, graph_index):
   global graph
   None
def plot_all_stacked_bar():
   global graph
   None

# rebuild the original data set from the buckerized one;
# used in stats_* on pre-buckerized data;
def original_X(values = None, counters = None, zipped_xy = None):
   from itertools import repeat
   if not zipped_xy:
      # not zipped_xy computed yet; let's do it here;
      zipped_xy = [(x, y) for x, y in zip(values, counters)]
   tmp = []
   for x, y in zipped_xy:
      tmp += list(repeat(x, int(y)))
   return tmp

# ----------------------------------------------------------------------------------------------------------------------
# plots of histograms and properly reformatted time series by the distfit package;
def distfit_plot(graph, plot_data):
   from distfit import distfit
   dfit = distfit()
   fig, axs = plt.subplots(2, 2, figsize = (25, 10))
   # prepare the data;
   if "timeseries" == plot_data["graph-type"]:
      # as we will modifiy data if timeseries are passed, we need to do a deep copy in order to preserve the original ones;
      import copy
      plot = copy.deepcopy(plot_data)

      plot["all-data"][1:] = [sp[0 if plot["graph-type"] == "histogram" else 1] for data in plot["all-data"][1:] if len(sp := data.split(",")) >= 1 and "" != all(sp)]
      # equivalent explicit loop statement:
      #for i, data in enumerate(plot["all-data"][1:]):
      #   if len(sp := data.split(",")) >= 1 and ""!= all(sp):
      #      plot["all-data"][i + 1] = sp[0 if plot["graph-type"] == "histogram" else 1]
      plot["x"] = [float(x) for x in plot["all-data"][1:]]
      np_plot = np.array(plot["x"])
   elif "buckets" == plot_data["graph-type"]:
      plot = plot_data
      plot["x"] = plot_data["x"]
      plot["y"] = plot_data["y"]
      np_plot = np.array(original_X(X = plot["x"], Y = plot["y"]))
   else:
      plot = plot_data
      np_plot = np.array(plot["x"])

   dfit.fit_transform(np_plot)
   dfit.predict(np_plot)
   dfit.plot(chart = 'pdf', n_top = 10, ax = axs[0, 0])
   axs[0, 0].set_title("Prob. Density Function (PDF)", fontsize = "large", fontweight = "bold")
   axs[0, 0].set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axs[0, 0].autoscale(tight = True)
   plt.show(block = False)
   dfit.plot(chart = 'cdf', n_top = 10, ax = axs[0, 1])
   axs[0, 1].set_title("Prob. Cumulative Function (CDF)", fontsize = "large", fontweight = "bold")
   axs[0, 1].set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axs[0, 1].autoscale(tight = True)
   plt.show(block = False)
   dfit.plot_summary(ax = axs[1, 0])
   axs[1, 0].set_title("Comparative Residual Squared Sums (RSS)", fontsize = "large", fontweight = "bold")
   axs[1, 0].autoscale(tight = True)
   plt.show(block = False)
   dfit.qqplot(np_plot, ax = axs[1, 1])
   axs[1, 1].set_title("Quantile to Quantile (Q-Q) Plot", fontsize = "large", fontweight = "bold")
   axs[1, 1].autoscale(tight = True)
   if "title" in plot:
      plt.get_current_fig_manager().set_window_title(f'Statistical Analysis with distfit - {plot["title"]} - file {plot["file-name"]}')
      fig.suptitle(f'distfit - {plot["title"]}', fontsize = "x-large", fontweight = "bold")
   if not graph["save-no-file"]:
      save_graph((plot["save-as"][:pos] + "_stat-distfit" + plot["save-as"][pos:] if (pos := plot["save-as"].rfind(".")) > -1 else plot["save-as"] + "_stat-distfit") if "save-as" in plot else None, plot["save-as-format"], graph)
   print(f'{dfit.summary[["name", "score", "loc", "scale", "arg"]]}')
   if graph["is-interactive"]:
      plt.tight_layout()
      fig.canvas.flush_events()
      plt.show(block = True)

# ----------------------------------------------------------------------------------------------------------------------
# plots one graph per theoretical statistical distribution on top of the histogram and properly reformatted time series;
# distributions are: uniform, normal, exponential, Poisson, and gamma (one graph for typical values of 1, 2, 3, 4, 5, 10, 20, 50, 100;
def stat_plot(graph, plot_data):
   import statistics as stats
   from collections import Counter

   # prepare the data;
   if "timeseries" == plot_data["graph-type"]:
      # as we will modifiy data if timeseries are passed, we need to do a deep copy in order to preserve the original ones;
      import copy
      plot = copy.deepcopy(plot_data)
      plot["all-data"][1:] = [sp[1] for data in plot["all-data"][1:] if len(sp := data.split(",")) >= 1 and "" != all(sp)]
      plot["x"] = [float(x) for x in plot["all-data"][1:]]
   elif "buckets" == plot_data["graph-type"]:
      plot = plot_data
      plot["x"] = plot_data["x"]
      plot["y"] = plot_data["y"]
   else:
      plot = plot_data

   # compute the average and standard deviation of dataset; they are used later by the theoretical distributions;
   if "buckets" != plot["graph-type"]:
      avg = stats.fmean(plot["x"])
      stddev = stats.stdev(plot["x"], avg)
   else:
      zipped_xy = [(x, y) for x, y in zip(plot["x"], plot["y"])]
      avg = sum([y * x for x, y in zipped_xy]) / sum(plot["y"])
      stddev = math.pow(sum([y * math.pow((x - avg), 2) for x, y in zipped_xy]) / sum(plot["y"]), 0.5)
      message(f'{avg = }, {stddev = }')

   # save current grid status before changing to no grid by default;
   saved_grid = plt.rcParams['axes.grid']
   plt.rcParams['axes.grid'] = False

   pos_legend = (0.99, 0.94)

   nb_graphs = 6
   nb_cols = 2
   fig, axs = plt.subplots(nrows = (nb_graphs - 1) // nb_cols + 1, ncols = nb_cols, squeeze = False, figsize = (16 * nb_cols, 9 * nb_cols)) # / 2))
   suptitle = "\n".join(("Statistical Analysis", f'{plot["title"] if plot["title"] else ""}, file {plot["file-name"]}, mean μ={avg:.2f}, stddev σ={stddev:.2f}, nb points n={len(plot["x"])}'))
   plt.get_current_fig_manager().set_window_title(suptitle)
   fig.suptitle("Statistical Analysis", fontsize = "medium", fontweight = "bold")

   # plot the real data;
   plot_index = 0
   axis = axs[plot_index // nb_cols, plot_index % nb_cols]
   if "buckets" == plot["graph-type"]:
      do_plot_individual_bar(graph, plot, axis)
      minx = plot["min-x"]; maxx = plot["max-x"]
   else:
      values = do_plot_individual_histogram(graph, plot, axis)
      minx = min(values[1]); maxx = max(values[1])

   axis.set_title("trying a normal distribution", fontsize = "large", fontweight = "bold")
   axis.legend(loc = 'upper right')
   # don't plot the vertical bar outside the plot area or the margins will be restablished to fit the data;
   if minx <= avg + stddev <= maxx:
      axis.axvline(x = avg + stddev, color = 'green', linestyle = "dashed", label = "mean")
   if minx <= avg - stddev <= maxx:
      axis.axvline(x = avg - stddev, color = 'green', linestyle = "dashed", label = "mean")
   if minx <= 2 * avg + stddev <= maxx:
      axis.axvline(x = avg + 2 * stddev, color = 'green', linestyle = "dashdot", label = "mean")
   if minx <= avg - 2 * stddev <= maxx:
      axis.axvline(x = avg - 2 * stddev, color = 'green', linestyle = "dashdot", label = "mean")
   if minx <= avg + 3 * stddev <= maxx:
      axis.axvline(x = avg + 3 * stddev, color = 'green', linestyle = "dotted", label = "mean")
   if minx <= avg - 3 * stddev <= maxx:
      axis.axvline(x = avg - 3 * stddev, color = 'green', linestyle = "dotted", label = "mean")
   if minx <= avg <= maxx:
      axis.axvline(x = avg, color = 'green', label = "mean")

   # plot the theoretical normal distribution N(avg, stdev);
   if "buckets" != plot["graph-type"]:
      x_values = values[1]
   else:
      x_values = plot["x"]
   a = 1.0/(stddev * math.sqrt(math.tau))
   y_values_theoretical = [a * math.exp(-0.5 * math.pow((x - avg)/stddev, 2)) for x in x_values]
   axis2 =  axis.twinx()
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis2.plot(x_values, y_values_theoretical, color = 'blue', label = f"theoretical gaussian distribution\n{avg=}, {stddev=}"), hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, X = x_values, Y = y_values_theoretical):
         indx = int(sel.index)
         sel.annotation.set(text = f'({X[indx]}, {Y[indx]:.2f})')
   else:
      axis2.plot(x_values, y_values_theoretical, color = 'blue', label = f"theoretical gaussian distribution\n{avg=}, {stddev=}")
   axis2.autoscale(tight = True)
   axis2.set_axis_off()
   legend = "\n".join(("theoretical gaussian distribution", f'mean μ={avg:.2f}, stddev σ={stddev:.2f}',
                      f'68.3% events occur between {avg - stddev if avg - stddev >= minx else minx:.2f} and {avg + stddev if avg + stddev <= maxx else maxx:.2f}',
                      f'95.5% between {avg - 2 * stddev if avg - 2 * stddev >= minx else minx:.2f} and {avg + 2 * stddev if avg + 2 * stddev <= maxx else maxx:.2f}',
                      f'99.8% between {avg - 3 * stddev if avg - 3 * stddev >= minx else minx:.2f} and {avg + 3 * stddev if avg + 3 * stddev <= maxx else maxx:.2f}'))
   axis2.text(*pos_legend, legend, color = 'blue', transform = axis.transAxes, fontsize = 9, verticalalignment = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'round', linewidth = 2, edgecolor = 'black', facecolor = 'lightgreen', alpha = 0.5))

   # Poisson distribution;
   from scipy.stats import poisson
   plot_index += 1
   axis = axs[plot_index // nb_cols, plot_index % nb_cols]
   if "buckets" == plot["graph-type"]:
      do_plot_individual_bar(graph, plot, axis)
   else:
      values = do_plot_individual_histogram(graph, plot, axis)
   axis.set_title("trying a Poisson distribution", fontsize = "large", fontweight = "bold")
   axis.legend()
   if minx <= avg <= maxx:
      axis.axvline(x = avg, color = 'green', label = "mean")
   # plot the theoretical Poisson distribution poisson(avg);
   if "buckets" != plot["graph-type"]:
      x_values = [int(x) for x in values[1]]
   else:
      x_values = plot["x"]
   # Poisson distribution is discrete; so, don't plot the continuous curve but single points instead;
   # Poisson distributions are very close to normal ones N(μ=λ, σ=sqr(λ) when the mean λ > 20, and the theoretical curve shows it;
   #y_values_theoretical = [math.pow(avg, x) * math.exp(-avg)/math.factorial(x) for x in x_values]
   # or use  axis2.hist(poisson.rvs(mu = avg, size = len(x_values)), color = 'blue');
   # let's use logarithms to prevent float overflows;
   y_values_theoretical = [math.exp(x * math.log(avg) + (-avg - math.log(math.factorial(x)))) for x in x_values]
   axis2 = axis.twinx()
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis2.plot(x_values, y_values_theoretical, color = 'blue', marker = 'o'), hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, X = x_values, Y = y_values_theoretical):
         indx = int(sel.index)
         sel.annotation.set(text = f'({X[indx]}, {Y[indx]:.2f})')
   else:
      axis2.plot(x_values, y_values_theoretical, color = 'blue', marker = 'o')
   axis2.autoscale(tight = True)
   axis2.axis([min(x_values), max(x_values), min(y_values_theoretical), max(y_values_theoretical)])
   axis2.set_axis_off()

   # find the values for probability <= limit ;
   # same to poisson.ppf(q, mu, loc = 0);
   def find_prob(limit):
      cte = limit/math.pow(math.e, -avg)
      k = 0
      while True:
         s = 0
         for i in range(k + 1):
            s += math.pow(avg, i)/math.factorial(i)
            if s >= cte:
               break
         else:
            k += 1
            continue
         break
      return k

   prob_steps = (0.25, 0.50, 0.75, 0.90, 0.99)
   try:
      #prob = [find_prob(p) for p in prob_steps]
      prob = [poisson.ppf(q = p, mu = avg, loc = 0) for p in prob_steps]
      linestyles = ["solid", "dashed", "dashdot", "dotted"]
      linestyles_str = ["―――", "― ―", "―.―", "....."]
      legend = "\n".join(("theoretical Poisson distr.",
                          f'mean λ={avg:.2f}, (stddev={math.pow(avg, 0.5):.2f})'))
      for i, p in enumerate(prob):
         if minx <= p <= maxx:
            axis.axvline(x = p, color = 'green', linestyle = linestyles[i], label = "mean")
         legend = legend + "\n" + linestyles_str[i] + f'{100 * prob_steps[i]:.2f}% not after {p:.2f}'
      axis2.text(*pos_legend, legend, color = 'blue', transform = axis.transAxes, fontsize = 10, verticalalignment = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'round', linewidth = 2, edgecolor = 'black', facecolor = 'lightgreen', alpha=0.5))
   except:
      # ignore this graph;
      message("math overflow, skipping computation of Poisson quartiles")

   # exponential distribution;
   plot_index += 1
   axis = axs[plot_index // nb_cols, plot_index % nb_cols]
   avg_exp = 1.0 / avg
   if "buckets" == plot["graph-type"]:
      do_plot_individual_bar(graph, plot, axis)
      y_values_theoretical = [avg_exp * math.exp(-avg_exp * x) for x in plot["x"] if x > 0]
      x_values = plot["x"]
   else:
      values = do_plot_individual_histogram(graph, plot, axis)
      x_values = values[1]
      y_values_theoretical = [avg_exp * math.exp(-avg_exp * x) for x in x_values if x > 0]
   axis.set_title("trying an exponential distribution", fontsize = "large", fontweight = "bold")
   axis.legend()
   if minx <= avg <= maxx:
      axis.axvline(x = avg, color = 'green', label = "mean")
   if minx <= 2 * avg <= maxx:
      axis.axvline(x = 2 * avg, color = 'green', linestyle = "dashed", label = "mean")
   if minx <= 3 * avg <= maxx:
      axis.axvline(x = 3 * avg, color = 'green', linestyle = "dashdot", label = "mean")
   if minx <= 4 * avg <= maxx:
      axis.axvline(x = 4 * avg, color = 'green', linestyle = "dotted", label = "mean")
   # plot the theoretical exponential distribution expon(av);
   axis2 = axis.twinx()
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis2.plot([x for x in x_values if x > 0], y_values_theoretical, color = 'blue'), hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, X = [x for x in x_values if x > 0], Y = y_values_theoretical):
         indx = int(sel.index)
         sel.annotation.set(text = f'({X[indx]}, {Y[indx]:.2f})')
   else:
      axis2.plot([x for x in x_values if x > 0], y_values_theoretical, color = 'blue')
   axis2.set_axis_off()
   legend =  "\n".join(("theoretical  exponential distr.",
                        f'rate λ={avg_exp:.2f} (stddev={avg_exp:.2f})',
                        f'63.2% events occur before {avg:.2f}',
                        f'{100 * (1 - math.exp(-avg_exp * 2 * avg)):.1f}%  events occur before {2 * avg:.2f}',
                        f'{100 * (1 - math.exp(-avg_exp * 3 * avg)):.1f}% events occur before {3 * avg:.2f}',
                        f'{100 * (1 - math.exp(-avg_exp * 4 * avg)):.1f}% events occur before {5 * avg:.2f}'
                        ))
   axis2.text(*pos_legend, legend, color = 'blue', transform = axis.transAxes, fontsize = 10, verticalalignment = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'round', linewidth = 2, edgecolor = 'black', facecolor = 'lightgreen', alpha=0.5))

   # uniform distribution;
   # plot the real-data histogram as the reference;
   plot_index += 1
   axis = axs[plot_index // nb_cols, plot_index % nb_cols]

   if "buckets" == plot["graph-type"]:
      do_plot_individual_bar(graph, plot, axis)
      x_values = plot["x"]
      avg_uniform = stats.fmean(plot["y"])
      y_values_theoretical = [avg_uniform for x in x_values]
   else:
      values = do_plot_individual_histogram(graph, plot, axis)
      # plot the theoretical uniform distribution U(avg_uniform);
      avg_uniform = sum(values[0]) / len(values[0])
      x_values = values[1]
      y_values_theoretical = [avg_uniform for x in x_values]

   axis.set_title("trying an uniform distribution", fontsize = "large", fontweight = "bold")
   axis.legend()

   axis.plot(x_values, y_values_theoretical, color = 'blue')
   legend =  "\n".join(("theoretical uniform distr.", f'frequency mean={avg_uniform:.2f}'))
   axis.text(*pos_legend, legend, color = 'blue', transform = axis.transAxes, fontsize = 10, verticalalignment = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'round', linewidth = 2, edgecolor = 'black', facecolor = 'lightgreen', alpha=0.5))

   # plot the quartiles;
   plot_index += 1
   # plot the real-data histogram as the reference;
   axis = axs[plot_index // nb_cols, plot_index % nb_cols]
   if "buckets" == plot["graph-type"]:
      do_plot_individual_bar(graph, plot, axis)
      quartiles = stats.quantiles(original_X(zipped_xy = zipped_xy), n = 4) #, method = "exclusive")
   else:
      do_plot_individual_histogram(graph, plot, axis)
      quartiles = stats.quantiles(plot["x"], n = 4)
   axis.set_title("quartiles", fontsize = "large", fontweight = "bold")
   axis.legend(loc = 'upper right')
   legend = "Quartiles"
   for i, q in enumerate(quartiles):
      q_info = f'{i + 1}{("st - 25%" if 0 == i else "nd - 50%" if 1 == i else "rd - 75%" if 2 == i else "th 100%") + " quartile at "}<= {q:.2f}'
      if minx <= q <= maxx:
         axis.axvline(x = q , color = 'green', label = q_info)
      legend = "\n".join((legend, q_info))
   axis.text(*pos_legend, legend, color = 'blue', transform = axis.transAxes, fontsize = 10, verticalalignment = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'round', linewidth = 2, edgecolor = 'black', facecolor = 'lightgreen', alpha=0.5))

   # cumulative distribution fonction;
   plot_index += 1
   axis = axs[plot_index // nb_cols, plot_index % nb_cols]
   axis.set_title("Cumulative Distribution Function (CDF)", fontsize = "large", fontweight = "bold")
   axis.legend(loc = 'upper right')
   axis.set_xlabel(plot["x-axis-label"], fontsize = "small", fontweight = "bold")
   axis.set_ylabel("probability", fontsize = "small", fontweight = "bold")

   if "buckets" == plot["graph-type"]:
      quartiles = stats.quantiles(original_X(zipped_xy = zipped_xy), n = 4, method = "inclusive")
      # Counter() is equivalent to the buckerized data sets thus, we can use the data set directly;
      counter = zipped_xy
   else:
      quartiles = stats.quantiles(plot["x"], n = 4, method = "inclusive")
      counter = sorted(Counter(plot["x"]).items())
      x_values = sorted([x[0] for x in counter] + quartiles)
      tot = sum([x[1] for x in counter])
   x_values = [x[0] for x in counter]
   tot = sum([x[1] for x in counter])
   cdf = []
   [cdf.append(x[1] + (cdf[-1] if cdf else 0)) for x in counter]
   y_values = [x/tot for x in cdf]

   legend = "cdf"
   if graph["is-interactive"] and graph["crosshair-cursor"]:
      cursor = mplcursors.cursor(axis.plot(x_values, y_values, color = 'blue', label = legend), hover = mplcursors.HoverMode.Transient)
      @cursor.connect("add")
      def on_add(sel, x_values = x_values, y_values = y_values):
         sel.annotation.set(text = f'({x_values[int(sel.index)]:.2f}, {y_values[int(sel.index)]:.2f})')
   else:
      axis.plot(x_values, y_values, color = 'blue', label = legend)
   axis.set_xlim(left = 0.0)
   axis.legend()
   for i, q in enumerate(quartiles):
      q_info = f'{i + 1}{("st - 25%" if 0 == i else "nd - 50%" if 1 == i else "rd - 75%" if 2 == i else "th 100%") + " quartile at "}<= {q:.2f}'
      if minx <= q <= maxx:
         axis.axvline(x = q , color = 'green', label = q_info)
      legend = "\n".join((legend, q_info))
   axis.autoscale(tight = True)
   axis.text(*pos_legend, legend, color = 'blue', transform = axis.transAxes, fontsize = 10, verticalalignment = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'round', linewidth = 2, edgecolor = 'black', facecolor = 'lightgreen', alpha=0.5))

   # remove residual, unused axes in figure;
   plot_index += 1
   for _ in range(plot_index, ((nb_graphs - 1) // nb_cols + 1) * nb_cols):
      axs.flat[_].remove()

   if not graph["save-no-file"]:
      save_graph((plot["save-as"][0 : pos] + "_stat" + plot["save-as"][pos:] if (pos := plot["save-as"].rfind(".")) > -1 else plot["save-as"] + "_stat") if "save-as" in plot else None, plot["save-as-format"], graph)

   if graph["is-interactive"]:
      plt.tight_layout()
      fig.canvas.flush_events()
      plt.show(block = True)

   # restore former grid status;
   plt.rcParams['axes.grid'] = saved_grid

def check_limits(limits):
   # processing limits for non categorical data;
   # if a limit is not number, we ignore the limits altogether;
   l = [None, None]
   for i, limit in enumerate(limits):
      if limit != None:
         try:
            l[i] = float(limit)
         except:
            l[i] = limit
      else:
         l[i] = limit
   return l

# performs some initial data processing such as identifying the kind of data sets (numerical or categorical), converting data to float when applicable, limiting the data set to the optional provided range;
# as soon as a non-number data is found in a data set, in any axis, the data set is considered as categorical and the data will be used as-is, i.e. not be converted to float nor sorted; otherwise, it is considered a numerical data set with float as values in both X and Y axis, and the X-axis sorted in ascending order;
# if ranges are defined as minx...maxx, the X axis's initial data will be constrained to that range; it is recommended to defined a range when too many data are to be plotted;
# in order to simplify the code, formatting errors (e.g. in dates not following the supported DATE_FORMAT), exceptions are not processed and the program left crashing; this is not an issue in interactive use and can be changed if needed;
def prepare_data():
   def to_numbers(plot, data_index):
      is_categorical = False
      try:
         lplot = [float(sp[data_index]) for data in plot["all-data"][1:] if len(sp := data.split(",")) >= 1 and "" != all(sp)]
      except:
         # likely a conversion to float error;
         # let's assume these are categorical data and appropriately deal with them later;
         is_categorical = True
         lplot = [sp[data_index] for data in plot["all-data"][1:] if len(sp := data.split(",")) >= 1 and "" != all(sp)]
      return lplot, is_categorical

   # graph types to check are "timeseries", "histogram", "bar", "line", "scatter", "stem", "pie", "buckets";
   # plot["x"] and plot["y"] are extracted and appended to the graph global;
   for gr in graph["graphs"]:
      for plot in gr["plot"]:
         # process the x-axis data resulting in plot["x"];
         if "timeseries" == plot["graph-type"]:
            # conversion of the ISO timestamps to datetime using the format string DATE_FORMAT;
            plot["x"] = [datetime.strptime(sp[0], DATE_FORMAT) for data in plot["all-data"][1:] if len(sp := data.split(",")) > 1 and "" != all(sp)]
         elif "pie" == plot["graph-type"]:
            # x values are not always numbers;
            plot["x"] = [sp[0] for data in plot["all-data"][1:] if len(sp := data.split(",")) > 1 and "" != all(sp)]
            plot["exploded"] = [0.1 if l > 2 and "e" == sp[2] else 0.0 for data in plot["all-data"][1:] if (l := len(sp := data.split(","))) > 1 and "" != all(sp)]
            plot["is-categorical-in-x"] = True
         else:
            plot["x"], plot["is-categorical-in-x"] = to_numbers(plot, 0)
         message(f'{plot["is-categorical-in-x"] = }')

         # process the y-axis data, if applicable, resulting in plot["y"];
         if "histogram" != plot["graph-type"]:
            # histograms' data sets have no y-axis;
            plot["y"], plot["is-categorical-in-y"] = to_numbers(plot, 1)
         message(f'{plot["is-categorical-in-y"] = }')

         # process the ranges' bounds if any;
         # pies are processed later since they are always categorical;
         # X-axis range bounds;
         if "timeseries" == plot["graph-type"]:
            if plot["min-x"]:
               plot["min-x"] = datetime.strptime(plot["min-x"], DATE_FORMAT)
            if plot["max-x"]:
               plot["max-x"] = datetime.strptime(plot["max-x"], DATE_FORMAT)
         else:
            plot["min-x"], plot["max-x"] = check_limits((plot["min-x"], plot["max-x"]))

         # Y-axis range bounds;
         if "histogram" == plot["graph-type"]:
            # histograms' data sets for don't have y values;
            plot["is-categorical-in-y"] = False
         else:
            plot["min-y"], plot["max-y"] = check_limits((plot["min-y"], plot["max-y"]) )

         # now, restrict the X-axis and Y_axis data as specified by the range bounds;
         # axis ranging must be synchronized, i.e. both axis must have same length and the 1-to-1 correspondance must be respected;
         if "histogram" == plot["graph-type"]:
            tmp = sorted(plot["x"])
            if plot["min-x"]:
               for i, v in enumerate(tmp):
                  if v >= plot["min-x"]:
                     left = i
                     break
               else:
                  left = 0
            else:
               left = 0
               plot["min-x"] = tmp[0]

            if plot["max-x"]:
               for i, v in reversed(list(enumerate(tmp))):
                  if v <= plot["max-x"]:
                     right = i
                     break
               else:
                  right = len(tmp)
            else:
               right = len(tmp)
               plot["max-x"] = tmp[-1]

            # final selection;
            plot["x"] = tmp[left:right]
         else:
            # X-axis ranging;
            if "timeseries" == plot["graph-type"]:
               tmp = sorted([(datetime.strptime(sp[0], DATE_FORMAT), float(sp[1]) if not plot["is-categorical-in-y"] else sp[1])
                             for data in plot["all-data"][1:] if len(sp := data.split(",")) > 1 and "" != all(sp)],
                            key = lambda p: p[0])
               plot["x"] = [c[0] for c in tmp]
            elif not plot["is-categorical-in-x"]:
               tmp = sorted([(float(sp[0]) if not plot["is-categorical-in-x"] else sp[0], float(sp[1]) if not plot["is-categorical-in-y"] else sp[1])
                             for data in plot["all-data"][1:] if len(sp := data.split(",")) > 1 and "" != all(sp)],
                            key = lambda p: p[0])
               plot["x"] = [c[0] for c in tmp]
            if plot["min-x"]:
               for i, v in enumerate(plot["x"]):
                  if v >= plot["min-x"]:
                     left = i
                     break
               else:
                  left = 0
            else:
               left = 0
               plot["min-x"] = plot["x"][0]

            if plot["max-x"]:
               #for i, v in reversed(list(enumerate(plot["x"]))):
               for i, v in enumerate(plot["x"]):
                  #if v <= plot["max-x"]:
                  #   right = i + 1
                  if v > plot["max-x"]:
                     right = i
                     break
               else:
                  right = len(plot["x"])
            else:
               right = len(plot["x"])
               plot["max-x"] = plot["x"][-1]

            # final selection;
            if "timeseries" == plot["graph-type"]:
               tmp = tmp[left:right]
               plot["x"] = [c[0] for c in tmp]
               plot["y"] = [float(c[1]) if not plot["is-categorical-in-y"] else c[1] for c in tmp]
            elif not plot["is-categorical-in-x"]:
               tmp = tmp[left:right]
               plot["x"] = [float(c[0]) for c in tmp]
               if not plot["is-categorical-in-y"]:
                  plot["y"] = [float(c[1]) for c in tmp]
               else:
                  plot["y"] = [c[1] for c in tmp]
            else:
               plot["x"] = plot["x"][left:right]
               plot["y"] = plot["y"][left:right]

            if not plot["is-categorical-in-x"] and "timeseries" != plot["graph-type"]:
               if plot["min-x"]:
                  plot["min-x"] = float(plot["min-x"])
               if plot["max-x"]:
                  plot["max-x"] = float(plot["max-x"])

            # Y-axis ranging;
            if not plot["is-categorical-in-y"]:
               if plot["min-y"]:
                  plot["min-y"] = float(plot["min-y"])
               if plot["max-y"]:
                  plot["max-y"] = float(plot["max-y"])
            if plot["min-y"]:
               # to preserve the length, use the placeholder np.nan, which is understood by matplotlib;
               plot["y"] = [c if c >= plot["min-y"] else np.nan for c in plot["y"]]
            else:
               plot["min-y"] = min(plot["y"]) if not plot["is-categorical-in-y"] else plot["y"][0]
            if plot["max-y"]:
               plot["y"] = [c if c <= plot["max-y"] else np.nan for c in plot["y"]]
            else:
               plot["max-y"] = max(plot["y"]) if not plot["is-categorical-in-y"] else plot["y"][-1]

 # ----------------------------------------------------------------------------------------------------------------------
# main;
if __name__ == '__main__':
   graph = {}
   graph["supertitle"], graph["input-files-group"], graph["titles"], graph["pie-geometries"], graph["save-as-list"], graph["save-as-format-list"], graph["nb-rows"], graph["nb-cols"], graph["save-as"], graph["b-replace"], graph["save-as-format"], graph["crosshair-cursor"], graph["is-interactive"], logging, graph["save-no-file"], graph["no-stats"], graph["no-distfit"], graph["stats-only"], graph["distfit-only"] = _ if (_ := parse_ok()) is not None else exit(1)
   if graph["titles"]:
      graph["titles"] = graph["titles"].split(",")
   show_params(graph, which_ones = ("global",))
   graph["graphs"] = []
   # normalize save-to-file parameters;
   for graph_index, plot_files in enumerate(graph["input-files-group"]):
      graph["graphs"].append({})
      graph["graphs"][graph_index]["save-as"] = _[graph_index] if graph["save-as-list"] and len(_ := graph["save-as-list"].split(",")) > graph_index else None
      graph["graphs"][graph_index]["save-as-format"] = _[graph_index] if graph["save-as-format-list"] and len(_ := graph["save-as-format-list"].split(",")) > graph_index and _[graph_index] in FORMAT_CHOICES else DEFAULT_FILE_FORMAT
      graph["graphs"][graph_index]["save-as"], graph["graphs"][graph_index]["b-replace"], graph["graphs"][graph_index]["save-as-format"] = saving_logic(graph["graphs"][graph_index]["save-as"], graph["graphs"][graph_index]["save-as-format"])
      show_params(graph["graphs"][graph_index], which_ones = ("groups,"))
      graph["graphs"][graph_index]["plot"] = []
      current_plot = None
      for plot_index, f in enumerate(plot_files[0].split(",")):
         # read in the graph parameters and its data from files;
         with open(f, "r") as g:
            plot = {}
            # ingest the file but skip empty and commented out lines;
            plot["all-data"] = [l for l in g.read().split("\n") if l and not l.startswith("#")]
            tmp = plot["all-data"][0].split("|")
            if current_plot and tmp[0] != current_plot:
               # plot types in same group must be identical;
               print(f"Warning: plot type {tmp[0]} is different from current group's plot type {current_plot}, ignoring file {f}")
               continue
            elif not tmp[0] in GRAPH_TYPES:
               print(f'unsupported graph type {tmp[0]}; it must be one of {GRAPH_TYPES}\nAborting ...')
               exit(1)
            else:
               plot["graph-type"] = current_plot = tmp[0]
            plot["is-categorical-in-x"] = False
            plot["is-categorical-in-y"] = False
            plot["file-name"] = f
            plot["title"] = tmp[1]
            plot["x-axis-label"] = tmp[2]
            plot["y-axis-label"] = tmp[3]
            plot["min-x"] = tmp[4]
            plot["max-x"] = tmp[5]
            plot["min-y"] = tmp[6]
            plot["max-y"] = tmp[7]
            plot["legends"] = tmp[8]
            plot["save-as"] = tmp[9]
            plot["save-as-format"] = tmp[10]
            plot["save-as"], plot["b-replace"], plot["save-as-format"] = saving_logic(plot["save-as"], plot["save-as-format"])
            if "histogram" == plot["graph-type"] and tmp[11]:
               plot["nb-bins"] = int(tmp[11])
            graph["graphs"][graph_index]["plot"].append(plot)
         show_params(graph, which_ones = ("dataset",), data = plot)
   show_params(graph, which_ones = ("raw",))

   prepare_data()

   # available styles are:
   #plt.style.available = ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
   # following 2 styles are pretty;
   #plt.style.use('seaborn-bright')
   plt.style.use('ggplot')

   # supported back-ends are: ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
   #matplotlib.use('QtCairo')
   # some of them - all those without GTKn, Qtn or Tk prefixes - are not interactive, e.g. 'agg', 'pdf', etc;
   # to avoid the error "ICE default IO error handler doing an exit(), pid = ..., errno = 32", tried Qt5Agg, QtAgg, TkAgg, QtCairo, Qt5Cairo, TkCairo but to no avail;
   # the above error seems related to the underlying X-Windows system, not matplotlib, and does not occur under MsWindows;
   matplotlib.use('TkAgg')

   # general library's default settings;
   plt.rcParams["figure.autolayout"] = True

   if not (graph["stats-only"] or graph["distfit-only"]):
      # functions to plot each file in its own graph and figure;
      individual_switches = {"timeseries": plot_individual_timeseries, "histogram": plot_individual_histogram, "scatter": plot_individual_scatter, "stem": plot_individual_stem, "line": plot_individual_line, "bar": plot_individual_bar, "pie": plot_individual_pie, "buckets": plot_individual_bar, "stacked-bar": plot_individual_stacked_bar}
      # each file is plotted in its own graph and figure (i.e. page/window/image file);
      for gr in graph["graphs"]:
         for plot in gr["plot"]:
            individual_switches[plot["graph-type"]](plot)

   if not (graph["stats-only"] or graph["distfit-only"]):
      # functions to plot all a group's files in its own group and figure;
      group_switches = {"timeseries": plot_group_timeseries, "histogram": plot_group_histograms, "scatter": plot_group_scatters, "stem": plot_group_stems, "line": plot_group_lines, "bar": plot_group_bars, "pie": plot_group_pies, "buckets": plot_group_bars, "stacked-bar": plot_group_stacked_bar}
      # all the groups are plotted into their own graphs but into one single figure for each group;
      plot_index = 0
      for group_index, gr in enumerate(graph["graphs"]):
         if 1 == len(gr["plot"]):
            # no need to plot groups with one file only as it was already plot individually;
            continue
         # since all the files in a group have same graph type, use the first file's one to determine the group's graph type;
         if "pie" != gr["plot"][0]["graph-type"]:
            group_switches[gr["plot"][0]["graph-type"]](gr, group_index)
         else:
            # needed to access the pie group's geometry;
            group_switches[gr["plot"][0]["graph-type"]](gr, group_index, plot_index)
            plot_index += 1

   if not (graph["stats-only"] or graph["distfit-only"]):
      # functions to plot all the groups in the same figure, with each group in its own graph (i.e. axe);
      # as there is no goto in python, fake loop to allow the break statement and exit when there is no graph to plot;
      for i in range(1):
         all_switches = {"timeseries": do_plot_group_timeseries, "histogram": do_plot_group_histograms, "scatter": do_plot_group_scatters, "stem": do_plot_group_stems, "line": do_plot_group_lines, "bar": do_plot_group_bars, "buckets": do_plot_group_bars, "pie": None}
         # all the groups are plotted in their own graph except for pies because they cannot be grouped and must be plot in a separate group per pie;
         # if there are pies, the global geometry will be overridden to a default of n x DEFAULT_GRAPHS_PER_ROW to accommodate them;
         nb_graphs = 0
         if len(graph["graphs"]) > 1:
            # no need for the global page when there are no groups or one group only that was already precedently plot;
            for group_index, gr in enumerate(graph["graphs"]):
               # groups are homogenous, i.e. all their graphs have the same type, thus testing the first one only is enough;
               if "pie" == (gr["plot"][0]["graph-type"]):
                  nb_graphs += len(gr["plot"])
               elif len(gr["plot"]) > 1:
                  # only consider groups with more than one data set and skip single plot graphs;
                  nb_graphs += 1
            if nb_graphs <= 1:
               break
            if nb_graphs > graph["nb-rows"] * graph["nb-cols"]:
               # requested geometry is insufficient to accommodate all the actual graphs, keep the number of columns but adjust the number of rows so there are enough cells for all the graphs;
               nb_cols = nb_graphs if nb_graphs <= graph["nb-cols"] else  graph["nb-cols"]
               nb_rows = (nb_graphs - 1) // nb_cols + 1
            else:
               # no pie charts, use the current geometry;
               nb_rows = graph["nb-rows"]
               nb_cols = graph["nb-cols"]
            fig, axs = plt.subplots(nrows = nb_rows, ncols = nb_cols, squeeze = False, figsize = (16 * nb_cols, 9 * nb_rows)) # / 2))
            plt.get_current_fig_manager().set_window_title(graph["supertitle"])
            fig.suptitle(graph["supertitle"], fontsize = "x-large", fontweight = "bold")
            pie_axes_offset = 0
            skipped_groups = 0
            for group_index, gr in enumerate(graph["graphs"]):
               if "pie" != (graph_type := gr["plot"][0]["graph-type"]):
                  if 1 == len(gr["plot"]):
                     # skip single plot graphs here, i.e. groups with one data set only;
                     skipped_groups += 1
                     continue
                  axis = axs.flat[group_index + pie_axes_offset - skipped_groups]
                  all_switches[graph_type](gr, group_index, axis)
               else:
                  # pie graphs are special because they cannot be grouped;
                  # so, we will plot each pie in a group into its own axe;
                  for pie_index, pie_plot in enumerate(gr["plot"]):
                     axis = axs.flat[group_index + pie_axes_offset + pie_index - skipped_groups]
                     do_plot_individual_pie(pie_plot, axis)
                  pie_axes_offset += pie_index
            # remove unused axes in figure;
            for _ in range(group_index + pie_axes_offset - skipped_groups + 1, nb_rows * nb_cols):
               axs[_ // nb_cols, _ % nb_cols].remove()
            save_graph(graph["save-as"], graph["save-as-format"], graph)
            show_figure(graph, fig)

   from multiprocessing import Process, set_start_method
   # time-consuming operations are carried over in their own process so not to freeze displaying the faster graphs;
   # using "spawn" instead of the default "fork"for matplotlib to work, cf. https://pythonspeed.com/articles/python-multiprocessing/;
   # note that global variables seem not be copied from the parent process over to the children processes and therefore must be passed as parameters;
   # the clue to that behavior is syntax errors on undeclared global variables;
   set_start_method("spawn")
   running_processes = []

   if not graph["no-stats"]:
      # statistical analysis starts here;
      # the idea is for each dataset to generate a figure containing a grouped graph per tentative distribution containing the original data's histogram and a curve for the theoretical curve;
      # e.g. each graph contains the histogram of the dataset + a line plot of the theoretical distribution based on the computed average (the loc) and standard deviation (the scale);
      # the above graph is repeated for each of the most frequently used distributions: uniform, normal, exponential, and Poisson;
      # stats computing are quite slow for large data sets; let's try to optimize it;
      # default multiprocessing using 'fork' does not work here with matplotlib because of the error below:
      #   xcb] Unknown sequence number while processing queue
      #   [xcb] Most likely this is a multi-threaded client and XInitThreads has not been called
      #   [xcb] Aborting, sorry about that.
      #   python3: ../../src/xcb_io.c:260: poll_for_event: Assertion `!xcb_xlib_threads_sequence_lost' failed.
      #   Aborted (core dumped)
      # multiprocessing with 'spawn' works fine though;
      for gr in graph["graphs"]:
         for plot in gr["plot"]:
            if not plot["graph-type"] in ("histogram", "timeseries", "buckets"):
               # ignore irrelevant data set
               message(f'skipping stats for {plot["graph-type"]} chart requested in {plot["file-name"]}\nstats graphs are plotted for non categorical histograms and timeseries data sets only')
               continue
            elif not plot["is-categorical-in-y"]:
               p = Process(target = stat_plot, args = (graph, plot))
               running_processes.append(p)
               p.start()

   if not graph["no-distfit"]:
      # same but using the distfit package for even more (ca. 89) theoretical distributions;
      for gr in graph["graphs"]:
         for plot in gr["plot"]:
            if not plot["graph-type"] in ("histogram", "timeseries", "buckets"):
               # ignore irrelevant data set
               message(f'skipping stats for {plot["graph-type"]} chart requested in {plot["file-name"]}\ndistfit graphs are plotted for non categorical histograms and timeseries data sets only')
               continue
            elif not plot["is-categorical-in-y"]:
               p = Process(target = distfit_plot, args = (graph, plot))
               running_processes.append(p)
               p.start()

   if graph["is-interactive"]:
      input("Press <enter> to exit")
      for p in running_processes:
         p.terminate()
   else:
      # wait for the children processes to complete;
      for p in running_processes:
         p.join()

   plt.close()
