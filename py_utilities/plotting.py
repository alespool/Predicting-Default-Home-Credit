import plotly.express as px
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from py_utilities.statistics_inference import test_normality_and_qq
from py_utilities.model_fitting import ModelFitter
from plotly.subplots import make_subplots
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_curve, auc, silhouette_samples, silhouette_score
from scipy.stats import poisson
import statsmodels.api as sm
from plotly.offline import iplot
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter


def create_qq_hist(data, column_name: str):
    """
    Test the normality of a column using the Anderson-Darling test and create a QQ plot.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to test for normality.

    Returns:
        tuple: A tuple containing the QQ plot figure and the histogram figure.
    """
    # QQ Plot using Plotly
    column_data = data[column_name].dropna()

    qq_fig = px.scatter(
        x=stats.probplot(column_data, dist="norm", fit=False)[0],
        y=stats.probplot(column_data, dist="norm", fit=False)[1],
        labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
        title="QQ Plot for " + column_name,
    )

    qq_fig.add_trace(go.Scatter(
        x=qq_fig.data[0].x,
        y=qq_fig.data[0].x, 
        mode='lines',
        name='Theoretical Quantiles',
        line=dict(color='red', dash='dash')
    ))

    # Histogram using Plotly
    hist_fig = px.histogram(column_data, nbins=30, title="Histogram for " + column_name)
    qq_fig.update_traces(aspectratio=0.5, selector=dict(type='funnelarea'))


    hist_fig.update_layout(
        width=800,
        height=600
    )

    return qq_fig, hist_fig


class Plotter:
    def __init__(self):
        pass

    def seaborn_histogram(self, df, x_col: str, title: str):
        """
        Generates a histogram plot using Seaborn.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data.
            x_col (str): The name of the column to use as the x-axis.
            title (str): The title of the plot.
        Returns:
            None
        """
        sns.histplot(data=df, x=x_col, palette='husl', multiple="stack", element="bars")
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
        

    def plotly_barplot_comparison(self, df, x_col:str, color_col:str, title:str):
        """
        Creates a barplot comparison using Plotly, displaying the distribution
        of the `x_col` across different categories defined by `color_col`.
        The plot is labeled with a `title`, and axis labels are set according
        to the `x_col` and `color_col` values. The number of bins is fixed to 20.
        Customized color scheme, plot dimensions, and bar gaps are applied.

        Parameters:
        - df (DataFrame): The pandas DataFrame containing the data to plot.
        - x_col (str): The name of the column in `df` to be used as the x-axis.
        - color_col (str): The name of the column in `df` that defines the
                        color categories for the bars.
        - title (str): The title of the plot.

        Returns:
        - None: This function displays the plot and does not return any value.
        """
        fig = px.histogram(df, x=x_col, color=color_col, nbins=20,
                        title=title, 
                        labels={'count':'Frequency', 'age':'Age'}, 
                        color_discrete_sequence=px.colors.qualitative.Pastel[:3])

        fig.update_layout(
            xaxis_title= x_col.title(),
            yaxis_title= "Frequency",
            legend_title= color_col.title(),
            width=800,
            height=600,
            bargap=0.1,
            bargroupgap=0.2
        )

        fig.show()

    def plotly_age_gender_distribution(self, df):
        """
            A function to plot the age and gender distribution using Plotly.
            Takes stroke_df as input and does not return anything.
        """
        
        fig = make_subplots(
            rows=2, cols=2,subplot_titles=('','<b>Distribution Of Female Ages<b>','<b>Distribution Of Male Ages<b>','Residuals'),
            vertical_spacing=0.09,
            specs=[[{"type": "pie","rowspan": 2}       ,{"type": "histogram"}] ,
                [None                               ,{"type": "histogram"}]            ,                                      
                ]
        )

        fig.add_trace(
            go.Pie(values=df.gender.value_counts().values,labels=['<b>Female<b>','<b>Male<b>','<b>Other<b>'],hole=0.3,pull=[0,0.08,0.3],marker_colors=['pink','lightblue','green'],textposition='inside'),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=df.query('gender=="Female"').age,marker= dict(color='pink'),name='Female Ages'
            ),
            row=1, col=2
        )


        fig.add_trace(
            go.Histogram(
                x=df.query('gender=="Male"').age,marker= dict(color='lightblue'),name='Male Ages'
            ),
            row=2, col=2
        )


        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="<b>Age-Sex Infrence<b>",
        )

        fig.show()

    def plotly_correlation_heatmap(self, df, show_values=True):
        """
        Generates a correlation heatmap using Plotly based on the given DataFrame.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data for which the correlation heatmap is to be generated.

        Returns:
            None
        """
        corr = df.corr(numeric_only=True)

        fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale='RdBu_r', labels=dict(color="Correlation"))
        fig.update_layout(
            title="Correlation Heatmap",
            width=800,
            height=600,
            xaxis=dict(tickangle=-45),
            yaxis=dict(title=''),
            coloraxis_colorbar=dict(title="Correlation", tickvals=[-1, -0.5, 0, 0.5, 1]),
        )

        if show_values:
            for i in range(len(corr.columns)):
                for j in range(len(corr.index)):
                    value = corr.iloc[j, i]
                    # Choose text color based on the background for better readability
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    fig.add_annotation(
                        x=i, y=j,
                        text=str(round(value, 2)),
                        showarrow=False,
                        font=dict(color=text_color)
                    )

        fig.show()
        
        
    def plotly_stroke_distribution(self, stroke_df, variable_1:str = 'gender', variable_2:str = 'ever_married', target:str = 'stroke'):
        """
        Plots the distribution of strokes with gender and marriage using heatmap visualizations.
        
        Parameters:
        - self: the object instance
        - stroke_df: the dataframe containing stroke data
        
        Returns:
        - None
        """
        fig = plt.figure(figsize=(12,6),dpi = 100)
        gs = fig.add_gridspec(1,2)
        gs.update(wspace=0.25, hspace=0.5)

        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])

        fig.patch.set_facecolor('#f6f5f5')
        ax0.set_facecolor('#f6f5f5')
        ax1.set_facecolor('#f6f5f5')

        healthy = stroke_df[stroke_df['stroke']==0]
        stroke = stroke_df[stroke_df['stroke']==1]

        col1 = ["#4b4b4c","#ff7f0e"]
        colormap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", col1, N = 256)
        col2 = ["#4b4b4c","#e377c2"]
        colormap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", col2)

        stroke = pd.crosstab(stroke[variable_1],[stroke[variable_2]],normalize='index')
        no_stroke = pd.crosstab(healthy[variable_1],[healthy[variable_2]], normalize='index')

        sns.heatmap(ax=ax0, data=stroke, linewidths= 0,
                    square=True, cbar_kws={"orientation": "horizontal"}, cbar=False,linewidth=3, cmap = col1,annot=True, fmt='1.0%',annot_kws={"fontsize":14}, alpha = 0.9)

        sns.heatmap(ax=ax1, data=no_stroke[:-1], linewidths=0, 
                    square=True, cbar_kws={"orientation": "horizontal"}, cbar=False,linewidth=3, cmap = col2,annot=True, fmt='1.0%',annot_kws={"fontsize":14}, alpha = 0.9)

        ax0.text(0, -0.69, 'Distribution of Strokes with Gender & Marriage', {'font':'Serif', 'color':'black', 'weight':'bold','size':25})
        ax0.text(0, -0.34, 'It is clear that married people are having more strokes \ncompared to singles. Married males are mostly \naffecteed, followed by married females.', {'font':'Serif', 'color':'black','size':14}, alpha = 0.7)

        ax0.text(0,-0.1,'Stroke Percentage', {'font':'serif', 'color':"#1f77b4", 'size':20},alpha = 0.9)
        ax1.text(0,-0.1,'No Stroke Percentage', {'font':'serif', 'color':"#512b58", 'size':20}, alpha =0.9)

        ax0.axes.set_xticklabels(['Single', 'Married'], fontdict={'font':'serif', 'color':'black', 'size':16})
        ax1.axes.set_xticklabels(['Single', 'Married'], fontdict={'font':'serif', 'color':'black', 'size':16})

        ax0.axes.set_yticklabels(['Female', 'Male'], fontdict={'font':'serif', 'color':'black', 'size':16}, rotation=0)

        ax0.set_xlabel('')
        ax0.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.axes.get_yaxis().set_visible(False)
        fig.show()
        
    def seaborn_violin_plots(self, df, x:str):
        """
        Generates box plots for attributes in the given DataFrame using Seaborn.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data for which box plots are to be generated.

        Returns:
            None
        """
        n_cols = 3  # Adjust the number of columns
        plot_width = 8  # Adjust the width of each plot
        plot_height = 5  # Adjust the height of each plot

        n_rows = (len(df.columns) - 1) // n_cols + 1  # Calculate the number of rows

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(plot_width * n_cols, plot_height * n_rows)
        )
        axes = axes.flatten()

        for i, column in enumerate(df.columns[:-1]):  # Exclude the 'quality' column
            ax = axes[i]
            sns.violinplot(data=df, x=x, y=column, ax=ax, hue=x, split=True)
            ax.set_title(column)

            if column == 'bmi':
                ax.set_ylim(10, 60)
                
        for j in range(i + 1, n_cols * n_rows):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def seaborn_distribution_plots(self, df, background_color:str = "#fafafa"):
        """
        Generates distribution plots for attributes in the given DataFrame using Seaborn.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data for which distribution plots are to be generated.

        Returns:
            None
        """
        n_cols = 4  # Adjust the number of columns
        plot_width = 6  # Adjust the width of each plot
        plot_height = 4  # Adjust the height of each plot

        background_color = background_color

        n_rows = (len(df.columns) // n_cols) + 1  # Calculate the number of rows

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(plot_width * n_cols, plot_height * n_rows)
        )
        axes = axes.flatten()

        for i, column in enumerate(df.columns):
            ax = axes[i]
            sns.histplot(data=df, x=column, ax=ax )
            ax.set_title(column)
            ax.set_facecolor(background_color)

        for j in range(i + 1, n_cols * n_rows):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_qq_hist(self, group, column_name: str):
        """
        Generate a QQ plot and histogram for a given group and test if the data follows a normal
        distribution.

        Parameters:
            group (DataFrame): The group to analyze.
            column_name (str): The column name to make plots over.

        Returns:
            fig (plot): The QQ plot of the data.
            hist_fig (plot): The histogram of the data.
        """
        is_normal, p = test_normality_and_qq(group, column_name=column_name)

        fig, hist_fig = create_qq_hist(group, column_name=column_name)

        if is_normal:
            print("The data follows a normal distribution.")
            print(f"P-value: {p}")
        else:
            print("The data does not follow a normal distribution.")
            print(f"P-value: {p}")

        return fig, hist_fig
    
    def plot_confusion_matrix(self, confusion_matrix, class_labels, title="Confusion Matrix"):
        """
        Create a confusion matrix plot using Plotly.

        Parameters:
            confusion_matrix (list of lists): The confusion matrix.
            class_labels (list of str): Labels for classes.
            title (str): Title for the plot.

        Returns:
            None
        """
        z_text = [[str(y) for y in row] for row in confusion_matrix]

        fig = ff.create_annotated_heatmap(
            z=confusion_matrix,
            x=class_labels,
            y=class_labels,
            annotation_text=z_text,
            colorscale="Viridis",
        )

        fig.update_layout(
            title_text=f"<i><b>{title}</b></i>",
            xaxis=dict(title="Predicted value"),
            yaxis=dict(title="Real value", autorange="reversed"),
            width=800,
            height=300
        )

        fig.update_layout(margin=dict(t=50, l=200))
        fig["data"][0]["showscale"] = True
        fig.show()

    def plot_multiclass_roc_auc_curve(self, y_probs, y_test):
        """
        Plots the ROC-AUC curve for multiclass classification using Plotly.

        Args:
            y_probs (array): Predicted class probabilities for each class.
            y_test (array): True labels for each sample.

        Returns:
            None
        """
        y_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs)
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig = go.Figure()

        colors = ['blue', 'green', 'red']

        for i, color in zip(range(n_classes), colors):
            fig.add_trace(go.Scatter(
                x=fpr[i],
                y=tpr[i],
                mode='lines',
                name=f'Class {i} (AUC = {roc_auc[i]:.2f})',
                line=dict(color=color),
                fill='tonexty'
            ))

        fig.add_shape(
            type='line',
            x0=0, x1=1,
            y0=0, y1=1,
            line=dict(dash='dash')
        )

        fig.update_xaxes(title_text='False Positive Rate')
        fig.update_yaxes(title_text='True Positive Rate')

        fig.update_layout(
            title='ROC-AUC Curve for Multiclass Classification',
            width=800,
            height=600,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )

        fig.show()

    def show_regression_predictions(self, train_predictions, test_predictions):
        """
        Generates a plot to display the regression predictions for both the training and testing datasets.

        Parameters:
            train_predictions (list): A list of predicted values for the training dataset.
            test_predictions (list): A list of predicted values for the testing dataset.

        Returns:
            None
        """
        fig = go.Figure()

        train_trace = go.Scatter(x=np.arange(len(train_predictions)), y=train_predictions, mode='lines', name='Train', line=dict(color='blue', width=2))
        test_trace = go.Scatter(x=np.arange(len(test_predictions)), y=test_predictions, mode='lines', name='Test', line=dict(color='red', width=2))

        fig.add_trace(train_trace)
        fig.add_trace(test_trace)

        train_mean = sum(train_predictions) / len(train_predictions)
        test_mean = sum(test_predictions) / len(test_predictions)

        fig.add_shape(go.layout.Shape(type="line", x0=0, x1=len(train_predictions), y0=train_mean, y1=train_mean, line=dict(color="blue", width=2)))
        fig.add_shape(go.layout.Shape(type="line", x0=0, x1=len(test_predictions), y0=test_mean, y1=test_mean, line=dict(color="red", width=2)))

        fig.update_layout(
            title='Training and Testing Predictions',
            xaxis=dict(title='Sample Index'),
            yaxis=dict(title='Predicted Values'),
            width=800,
            height=600,
            showlegend=True
        )

        fig.show()

        return test_mean, train_mean

    def plot_coefficients(self, coefs, feature_names):
        """
        Generate a bar plot of the coefficients for each feature in a linear regression model.

        Parameters:
            coefs (array-like): The coefficients of the linear regression model.
            feature_names (array-like): The names of the features.

        Returns:
            fig (plotly.graph_objs._figure.Figure): The bar plot of the coefficients.
        """
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
        coef_df = coef_df.round({'Coefficient': 2}) 
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False) 

        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', text='Coefficient')
        fig.update_layout(
            title='Linear Regression Coefficients',
            xaxis_title='Coefficient Value',
            yaxis_title='Feature',
            showlegend=False,
            width=800,
            height=600,
            xaxis_range=[-5.5,5.5]
        )
        
        return fig
        
    def plot_features_importance(self, model, X_preprocessed, X_original, y):
        """
        Plots the feature importance of a model using permutation importance.

        Parameters:
            model: The trained model for which feature importance is to be plotted.
            X_preprocessed: The preprocessed input features used for prediction.
            X_original: The original input features before preprocessing.
            y: The target variable used for prediction.

        Returns:
            indices_to_remove: Indices of columns with importance less than or equal to 0.
        """
        
        importance = permutation_importance(model, X_preprocessed, y, scoring='recall')
        sorted_idx = importance.importances_mean.argsort()

        # Align X_preprocessed with X_original
        X_preprocessed_aligned = X_original.iloc[:, sorted_idx]

        fig = go.Figure(data=go.Bar(
            x=importance.importances_mean[sorted_idx],
            y=X_preprocessed_aligned.columns,  # Using aligned feature names
            orientation='h'
        ))

        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            width=1000,
            height=600,
            yaxis=dict(autorange="reversed")
        )

        fig.show()
        
        # Get indices of columns with importance less than or equal to 0
        indices_to_remove = sorted_idx[importance.importances_mean <= 0]
        
        return indices_to_remove

    def plot_explained_variance(self, explained_variance):
        """
        Generates a plot to visualize the cumulative explained variance ratio for principal components analysis (PCA).

        Parameters:
            explained_variance (list): A list of explained variances for each principal component.

        Returns:
            None
        """

        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio for PCA')
        plt.grid(True)
        plt.show()

    def plot_histograms_and_poisson(self, data_group_a, data_group_b, mean_home_goals, mean_away_goals):
        """
        Plot histograms and Poisson distributions.
        
        Args:
            data_group_a (array-like): The data for group A.
            data_group_b (array-like): The data for group B.
            mean_home_goals (float): The mean of home goals.
            mean_away_goals (float): The mean of away goals.
        
        Returns:
            None
        """
        # Plot histograms
        hist_group_a = go.Histogram(x=data_group_a, histnorm='probability', name='Home Goals', opacity=0.5)
        hist_group_b = go.Histogram(x=data_group_b, histnorm='probability', name='Away Goals', opacity=0.5)

        # Plot Poisson distributions
        x = list(range(max(max(data_group_a), max(data_group_b)) + 1))
        poisson_home_goals = go.Scatter(x=x, y=poisson.pmf(x, mean_home_goals), mode='lines+markers', name='Poisson (Home Goals)', marker=dict(color='blue'))
        poisson_away_goals = go.Scatter(x=x, y=poisson.pmf(x, mean_away_goals), mode='lines+markers', name='Poisson (Away Goals)', marker=dict(color='red'))

        # Layout settings
        layout = go.Layout(
            title='Poisson Distribution of Home and Away Goals',
            xaxis=dict(title='Number of Goals'),
            yaxis=dict(title='Probability'),
            legend=dict(x=0.7, y=0.9),
            barmode='overlay'
        )

        fig = go.Figure(data=[hist_group_a, hist_group_b, poisson_home_goals, poisson_away_goals], layout=layout)
        fig.show()

    def plot_precision_recall_vs_threshold_plotly(self, precisions, recalls, thresholds, threshold_recall=None, threshold_precision=None):
        """
        Generates a plot of precision and recall against threshold values using Plotly.
        
        Args:
            precisions (list): A list of precision values.
            recalls (list): A list of recall values.
            thresholds (list): A list of threshold values.
            threshold_recall (float, optional): The threshold value for 90% recall.
            threshold_precision (float, optional): The threshold value for 90% precision.
        """
        trace1 = go.Scatter(x=thresholds, y=precisions[:-1], mode='lines', name='Precision', line=dict(color='blue', dash='dash'))
        trace2 = go.Scatter(x=thresholds, y=recalls[:-1], mode='lines', name='Recall', line=dict(color='green'))
        
        data = [trace1, trace2]

        layout = go.Layout(
            title='Precision and Recall vs Threshold',
            xaxis=dict(title='Threshold'),
            yaxis=dict(title='Precision/Recall', range=[0, 1]),
            showlegend=False
        )

        fig = go.Figure(data=data, layout=layout)

        if threshold_recall is not None:
            fig.add_trace(go.Scatter(
                x=[threshold_recall, threshold_recall], y=[0, 1],
                mode='lines', name='Threshold for 90% recall',
                line=dict(color='red', dash='dot')
            ))
            
        if threshold_precision is not None:
            fig.add_trace(go.Scatter(
                x=[threshold_precision, threshold_precision], y=[0, 1],
                mode='lines', name='Threshold for 90% precision',
                line=dict(color='purple', dash='dot')
            ))

        fig.update_layout(
            width=800,
            height=600
        )

        iplot(fig)

    def plot_learning_curves(self, model, X, y, train_sizes, cv, scoring):
        """
        Generates learning curves plot using Plotly.

        Parameters:
            model (object): The machine learning model.
            X (array-like): The input features.
            y (array-like): The target variable.
            train_sizes (array-like): The sizes of the training set.
            cv (int or cross-validation generator): Determines the cross-validation splitting strategy.
            scoring (str or callable): The scoring method for evaluating the predictions.

        Returns:
            None

        This function generates learning curves using the provided machine learning model and data. It calculates the mean and standard deviation of the training set scores and the validation set scores. It then creates traces for the plot, including the lower bound, upper bound, and mean of the training scores, as well as the lower bound, upper bound, and mean of the cross-validation scores. Finally, it combines the traces and plots them using Plotly.
        """
                
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring  
        )
        
        # Calculate the mean and standard deviation for training set scores
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        
        # Calculate the mean and standard deviation for validation set scores
        validation_mean = validation_scores.mean(axis=1)
        validation_std = validation_scores.std(axis=1)
        
        # Create traces for the plot
        trace1 = go.Scatter(
            x=train_sizes, y=train_mean - train_std,
            mode='lines', name='Training score lower bound',
            line=dict(width=0.5, color='rgba(255, 0, 0, 0.2)'),
            fill=None
        )
        trace2 = go.Scatter(
            x=train_sizes, y=train_mean + train_std,
            mode='lines', name='Training score upper bound',
            line=dict(width=0.5, color='rgba(255, 0, 0, 0.2)'),
            fill='tonexty' # this fills the area between the two traces
        )
        trace3 = go.Scatter(
            x=train_sizes, y=train_mean,
            mode='lines+markers', name='Training score',
            line=dict(color='red')
        )
        trace4 = go.Scatter(
            x=train_sizes, y=validation_mean - validation_std,
            mode='lines', name='Cross-validation score lower bound',
            line=dict(width=0.5, color='rgba(0, 255, 0, 0.2)'),
            fill=None
        )
        trace5 = go.Scatter(
            x=train_sizes, y=validation_mean + validation_std,
            mode='lines', name='Cross-validation score upper bound',
            line=dict(width=0.5, color='rgba(0, 255, 0, 0.2)'),
            fill='tonexty' # this fills the area between the two traces
        )
        trace6 = go.Scatter(
            x=train_sizes, y=validation_mean,
            mode='lines+markers', name='Cross-validation score',
            line=dict(color='green')
        )
        
        fig = go.Figure()
        fig.add_traces([trace1, trace2, trace3, trace4, trace5, trace6])
        
        fig.update_layout(
            title='Learning Curves',
            xaxis=dict(title='Training set size'),
            yaxis=dict(title=f'{scoring.title()} Score'),
            showlegend=True
        )
        
        fig.show()
        
    def silhouette_plot(self, scaled_features, kmeans):
        """
        Plot the silhouette plot for the KMeans clustering on the given scaled features.

        Args:
            scaled_features: The scaled features used for clustering.
            kmeans: The KMeans clustering model.

        Returns:
            None
        """
        labels = kmeans.labels_
        n_clusters = 3

        sample_silhouette_values = silhouette_samples(scaled_features, labels)

        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(scaled_features) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        silhouette_avg = silhouette_score(scaled_features, labels)
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

        plt.show()