import numpy as np
import pyPLS
import seaborn as sns
import pandas as pd
import random
from scipy import interpolate
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import string
from cycler import cycler
import matplotlib as mpl

params = {'mathtext.default': 'regular', 
          'figure.facecolor': 'white', 
          'font.size': 20}
plt.rcParams.update(params)

def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_featurecols(df):
    """returna  list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_metacols(df)]


def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]

def log_transform(s):
    if s >= 0:
        return np.log(s + 1)
    if s < 0:
        return -np.log(abs(s) + 1)
    
def calc_eq_score_df_with_cv_optimized(fit_df, pred_df, max_ncp=10, basis_col='Metadata_control_type', basis='negcon', reference_col='Metadata_pert_iname'):
    results_df = get_metadata(pred_df)
    ess_df = get_metadata(pred_df)
    eqs = fit_df[fit_df[basis_col] != basis][reference_col].unique().tolist()
    q2_list = [] 

    for eq in eqs:
        work = fit_df[(fit_df[reference_col] == eq) | (fit_df[basis_col] == basis)]
        n_samples = work[work[reference_col] == eq].shape[0]
        cvfolds = max(2, n_samples)

        X = np.array(get_featuredata(work), dtype=float)
        Y = np.array(work[basis_col] != basis, dtype=float)

        best_q2 = -np.inf
        best_ncp = 1

        for ncp in range(1, max_ncp + 1):
            pls_model = pyPLS.pls(X, Y, ncp=ncp, cvfold=cvfolds)
            q2 = pls_model.Q2Y

            if q2 > best_q2:
                best_q2 = q2
                best_ncp = ncp

        pls_model = pyPLS.pls(X, Y, ncp=best_ncp, cvfold=cvfolds)
        pred_cv = pls_model.Yhatcv

        # Compute predictions for all rows
        all_pred_features = get_featuredata(pred_df)
        all_predictions, all_stats = pls_model.predict(np.array(all_pred_features, dtype=float), statistics=True)

        # Assign predictions using vectorized operations
        is_eq = (pred_df[reference_col] == eq) | (pred_df[basis_col] == basis)
        is_not_eq = ~is_eq
        results_df.loc[is_not_eq, f"{eq}_eq"] = all_predictions[is_not_eq]
        ess_df[f"{eq}_SSE"] = all_stats['ESS']/1000
        if is_eq.any():
            # Create a boolean mask that aligns with work.index
            mask = is_eq.reindex(work.index, fill_value=False)
            # Filter work.index and pred_cv
            filtered_work_index = work.index[mask]
            filtered_pred_cv = pred_cv[mask.values]
            # Assign the filtered pred_cv values to results_df
            results_df.loc[filtered_work_index, f"{eq}_eq"] = filtered_pred_cv

        q2_list.append(pls_model.Q2Y)

    q2_df = pd.DataFrame(data={'References': eqs, 'Q2': q2_list})
    return results_df, ess_df, q2_df

def knn_map_loocv(X, y, ks):
    # Convert y to numpy array if it's a Pandas Series
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Filter out 'negcon' if it exists in y
    if 'negcon' in np.unique(y).tolist():
        valid_indices = y != 'negcon'
        X_valid = X[valid_indices]
        y_valid = y[valid_indices]
    else:
        X_valid = X
        y_valid = y

    n_samples = X_valid.shape[0]

    precision_at_k_scores = {k: [] for k in ks}
    recall_at_k_scores = {k: [] for k in ks}

    # Leave-One-Out Cross-Validation
    for k in ks:
        y_pred = []
        for test_index in range(n_samples):
            # Creating training set by excluding the test data point
            X_train = np.delete(X_valid, test_index, axis=0)
            y_train = np.delete(y_valid, test_index)

            # The test data point
            X_test = X_valid[test_index].reshape(1, -1)

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            # Predicting using the KNN model
            y_pred.append(knn.predict(X_test))

        # Calculating precision and recall for the current test data point
        precision = precision_score(y_valid, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_valid, y_pred, average='macro')

        precision_at_k_scores[k].append(precision)
        recall_at_k_scores[k].append(recall)

    # Averaging precision and recall scores for each k
    avg_precision_at_k = {k: np.mean(precision_at_k_scores[k]) for k in ks}
    avg_recall_at_k = {k: np.mean(recall_at_k_scores[k]) for k in ks}

    # Creating DataFrame for precision and recall
    df_recall_prec = pd.DataFrame({'Precision': [avg_precision_at_k[k] for k in ks], 
                                   'Recall': [avg_recall_at_k[k] for k in ks], 
                                   'K': ks}, index=ks)

    return df_recall_prec


class ClusterPlot():
    def _get_interpolate_plot_(self, points, interpolate=False):
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices,0],
                           points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                           points[hull.vertices,1][0])
        if interpolate:
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull], 
                                            u=dist_along, s=0)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)
        if not interpolate:
            interp_x, interp_y = (x_hull, y_hull)
        return(interp_x, interp_y)

    def _get_lines_plot_(self, points):
        center = np.sum(points, axis=0)/len(points[:, 0])
        cen_x, cen_y = center
        x = [points[:, 0], cen_x*np.ones(len(points))]
        y = [points[:, 1], cen_y*np.ones(len(points))]
        linexy = (x, y)
        cenxy = (cen_x, cen_y)
        return (cenxy, linexy)
    
    def plot_fig_with_clusters(self, df, metadata_grouping, x_axis, y_axis, figure_size=(15, 10), markersize=15, add_df=None, plot_add=False, plot_add_circle=False, bw=False, ax=None):
        if bw:
            cm = plt.get_cmap('Greys')
            abcs = list(string.ascii_lowercase)
            abc_i = 0
        else:
            cm = plt.get_cmap('tab20')
        n_colors = len(df[metadata_grouping].unique())
        if add_df is not None:
            n_colors = len(df[metadata_grouping].unique()) + len(add_df[metadata_grouping].unique())
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[cm(1.*i/n_colors) for i in range(n_colors)])
        aa = 0.2
        if ax == None:
            fig, ax = plt.subplots()
        
        for pert, perts in df.groupby(metadata_grouping):
            points = perts[[x_axis, y_axis]].values
            interp_x, interp_y = self._get_interpolate_plot_(points)
            cenxy, linexy = self._get_lines_plot_(points)
            if not bw:
                ax.fill(interp_x, interp_y, alpha=aa)
                
                p = ax.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize)
                ax.plot(linexy[0], linexy[1], c=p[0].get_color(), alpha=aa*2)
                ax.plot(perts[x_axis].values, perts[y_axis].values, '.', label=pert, c=p[0].get_color(), markersize=markersize)
            elif bw:
                all_lines = [np.sqrt(x**2 + y**2) for x, y in zip(linexy[0]-cenxy[0], linexy[1]-cenxy[1])]
                longest_line = np.max(all_lines)
                plt.text(cenxy[0]-longest_line*0.8, cenxy[1]+longest_line, f"{abcs[abc_i]}")            
                plt.fill(interp_x, interp_y, alpha=aa, c='black')
                p = plt.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize, c='black')
                plt.plot(linexy[0], linexy[1], c='black', alpha=aa*2)
                plt.plot(perts[x_axis].values, perts[y_axis].values, '.', label=f"{abcs[abc_i]} - {pert}", c='black', markersize=markersize)
                abc_i+=1
        if add_df is not None and plot_add:
            for pert, perts in add_df.groupby(metadata_grouping):
                points = perts[[x_axis, y_axis]].values
                interp_x, interp_y = self._get_interpolate_plot_(points)
                cenxy, linexy = self._get_lines_plot_(points)
                if not bw:
                    plt.fill(interp_x, interp_y, alpha=aa)
                    p = plt.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize)
                    plt.plot(linexy[0], linexy[1], c=p[0].get_color(), alpha=aa*2)
                    plt.plot(perts[x_axis].values, perts[y_axis].values, '.', label=pert, c=p[0].get_color(), markersize=markersize)
                elif bw:
                    all_lines = [np.sqrt(x**2 + y**2) for x, y in zip(linexy[0]-cenxy[0], linexy[1]-cenxy[1])]
                    longest_line = np.max(all_lines)
                    plt.text(cenxy[0]-longest_line*0.8, cenxy[1]+longest_line, f"{abcs[abc_i]}")       
                    plt.fill(interp_x, interp_y, alpha=aa, c='black')
                    p = plt.plot(cenxy[0], cenxy[1], marker='^', markersize=markersize, c='black')
                    plt.plot(linexy[0], linexy[1], c='black', alpha=aa*2)
                    plt.plot(perts[x_axis].values, perts[y_axis].values, '.', label=f"{abcs[abc_i]} - {pert}", c='black', markersize=markersize)
                    abc_i+=1
                if plot_add_circle:
                    all_lines = [np.sqrt(x**2 + y**2) for x, y in zip(linexy[0]-cenxy[0], linexy[1]-cenxy[1])]
                    longest_line = np.max(all_lines)
                    if bw:
                        circ = plt.Circle(cenxy, longest_line*1.2, color='black', fill=False)
                    elif not bw:
                        circ = plt.Circle(cenxy, longest_line*1.2, color='red', fill=False)
                    plt.gca().add_patch(circ)
        ax.legend(loc=(1.05,0))
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        return plt.gca()

    def plot_fig_with_mean_clusters(self, df, metadata_grouping, x_axis, y_axis, figure_size=(15, 10)):
        cm = plt.get_cmap('tab20')
        n_colors = len(df[metadata_grouping].unique())
        mpl.rcParams['axes.prop_cycle'] = cycler(color=[cm(1.*i/n_colors) for i in range(n_colors)])
        aa = 0.2
        fig = plt.figure(figsize=figure_size)
        negcon_points = df[df[metadata_grouping] == 'DMSO'][[x_axis, y_axis]].values
        negcon_center = np.sum(negcon_points, axis=0)/len(negcon_points[:, 0])
        cen_dict = {}
        dist_dict = {}
        mean_dist_dict = {}
        for pert, perts in df.groupby(metadata_grouping):
            if pert == 'DMSO':
                p = plt.plot(negcon_center[0], negcon_center[1], marker='o')#, c='black')
                points = perts[[x_axis, y_axis]].values
                cenxy, linexy = self._get_lines_plot_(points)
                mean_dist_dict.update({pert:  np.mean(np.sqrt(((linexy[0][0]-cenxy[0]))**2 + (linexy[1][0]-cenxy[1])**2))})
                circ = plt.Circle(cenxy, mean_dist_dict[pert], color=p[0].get_color())
                plt.gca().add_patch(circ)
            else:
                points = perts[[x_axis, y_axis]].values
                cenxy, linexy = self._get_lines_plot_(points)
                cen_dict.update({pert: cenxy - negcon_center})
                dist_dict.update({pert:  np.sqrt(cen_dict[pert][0]**2 + cen_dict[pert][1]**2)})
                mean_dist_dict.update({pert:  np.mean(np.sqrt(((linexy[0][0]-cenxy[0]))**2 + (linexy[1][0]-cenxy[1])**2))})
                p = plt.plot(cenxy[0], cenxy[1], marker='o')
                circ = plt.Circle(cenxy, mean_dist_dict[pert], alpha=aa, color=p[0].get_color())
                plt.gca().add_patch(circ)
                plt.plot([negcon_center[0], cenxy[0]], [negcon_center[1], cenxy[1]], '-', label=pert, c=p[0].get_color())
        plt.legend(loc=(1.05,0))
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        return plt.gca(), mean_dist_dict, cen_dict

def create_grouped_scatter_plot(df):
    ## Very specific function for the prec_recall files
    # Set the size of the entire figure
    fig = plt.figure(figsize=(20, 16))  # Adjusted for better visibility
    font_size = 36
    # Unique values for faceting
    cell_lines = df['Cell Line'].unique()
    times = df['Time'].unique()

    # Calculate the number of subplots needed
    num_rows = len(cell_lines)
    num_cols = len(times)
    # Find global max for 'Recall' and 'Precision' to set unified axes limits
    df['size'] = df['K'].values*2 + 120

    # Create a subplot for each combination of Cell Line and Time
    for i, cell_line in enumerate(cell_lines):
        for j, time in enumerate(times):
            ax = plt.subplot(num_rows, num_cols, i * num_cols + j + 1)

            # Filter data for each subplot
            filtered_df = df[(df['Cell Line'] == cell_line) & (df['Time'] == time)]

            size_range = (3000, 5000)  # Min and max sizes of the markers

            # Create scatter plot
            palette = ["#000000", "#2ca02c", "#ff7f0e"]
            scatter = sns.scatterplot(data=filtered_df, x='Recall', y='Precision', hue='Modality', size='size', alpha=0.7, sizes=size_range, linewidth=2, style='Modality')

            # Set the same x and y limits for all plots
            plt.xlim(0, 0.75)
            plt.ylim(0, 0.75)
            # Set labels for columns and rows
            if i == 0:
                plt.title(f'Time: {time}', fontsize=font_size)

            plt.xlabel('Recall' if i == num_rows - 1 else '', fontdict=dict(size=font_size))
            plt.ylabel('Precision' if j == 0 else '', fontdict=dict(size=font_size))
            ## Set ticklabel fontsize
            plt.setp(ax.get_xticklabels(), fontsize=24)
            plt.setp(ax.get_yticklabels(), fontsize=24)
            ax.grid(visible=True, linestyle='--', linewidth=1.5, color='w')
            ax.set_axisbelow(True)
            ax.legend().remove()
            ax.set_facecolor('#e5ecf6')
            # Manually placing the 'Cell Line' label on the right side of each row
            if j == 1 and i == 0:
                # for major ticks
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
            elif j == 1 and i == 0:
                # for major ticks
                plt.setp(ax.get_yticklabels(), visible=False)
            elif j == 0 and i == 0:
                # for major ticks
                plt.setp(ax.get_xticklabels(), visible=False)
            elif j == 1 and i == 1:
                # for major ticks
                plt.setp(ax.get_yticklabels(), visible=False)
            if j == num_cols - 1:
                ax.text(1.03, 0.5, f'Cell Line: {cell_line}', verticalalignment='center', horizontalalignment='center', transform=ax.transAxes, fontsize=font_size, rotation=270)

    # Create a custom legend for Modality
    handles, labels = scatter.get_legend_handles_labels()
    fig.legend([handles[1], handles[3], handles[2]], [labels[1], labels[3], labels[2]], loc='lower left', bbox_to_anchor=(0.2, -0.06), fancybox=False, shadow=False, frameon=False, ncol=len(labels[1:4]), fontsize=font_size, markerscale=6)
    plt.tight_layout()
    return plt