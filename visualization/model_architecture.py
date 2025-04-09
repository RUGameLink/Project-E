import io
import sys
import os

# Import TensorFlow with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    def load_model(*args, **kwargs):
        print("TensorFlow не установлен. Используйте Python 3.10 или 3.11, или установите tensorflow>=2.16.1 вручную.")
        return None

# Import visualization libraries with fallbacks
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    
try:
    import numpy as np
    
    # Define np functions to use regardless of whether np is None later
    def prod(values):
        return np.prod(values)
    def linspace(start, stop, num):
        return np.linspace(start, stop, num)
    def array_linspace(start, stop, num, dtype=None):
        return np.linspace(start, stop, num, dtype=dtype)
except ImportError:
    np = None
    # Fallback implementations
    def prod(values):
        result = 1
        for v in values:
            result *= v
        return result
    def linspace(start, stop, num):
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]
    def array_linspace(start, stop, num, dtype=None):
        values = linspace(start, stop, num)
        if dtype == int:
            return [int(v) for v in values]
        return values
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Create Figure class fallback if go is None
if go is None:
    class FigureFallback:
        def __init__(self):
            self.shapes = []
            self.annotations = []
        
        def add_shape(self, **kwargs):
            self.shapes.append(kwargs)
            return self
        
        def add_annotation(self, **kwargs):
            self.annotations.append(kwargs)
            return self
        
        def update_layout(self, **kwargs):
            self.layout = kwargs
            return self
    
    class TableFallback:
        def __init__(self, **kwargs):
            self.data = kwargs
    
    class GoFallback:
        def Figure(self, data=None):
            return FigureFallback()
        
        def Table(self, **kwargs):
            return TableFallback(**kwargs)
    
    go = GoFallback()

def get_layer_dimensions(model):
    """Calculate dimensions for each layer to visualize network architecture."""
    if model is None:
        return []
        
    layer_dims = []
    
    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        layer_type = layer.__class__.__name__
        
        # Get layer units/filters based on layer type
        if hasattr(layer, 'units'):
            units = layer.units
        elif hasattr(layer, 'filters'):
            units = layer.filters
        elif layer_type in ['InputLayer', 'Reshape', 'Flatten']:
            # For input layers, use the output shape
            units = prod(layer.output_shape[1:])
        else:
            # Default for other layer types
            units = 1
        
        layer_dims.append({
            'index': i,
            'name': layer.name,
            'type': layer_type,
            'units': int(units),
            'connections': []
        })
    
    # Add connections between layers
    for i in range(len(layer_dims) - 1):
        layer_dims[i]['connections'].append(i + 1)
    
    return layer_dims

def plot_model_architecture(model):
    """Create a plotly visualization of model architecture."""
    if model is None or go is None:
        print("Cannot plot model architecture: model or plotly is None")
        return None
        
    # Get layer dimensions
    layer_dims = get_layer_dimensions(model)
    
    if not layer_dims:
        return None
        
    # Create figure
    fig = go.Figure()
    
    # Calculate max height for scaling
    max_units = max([layer['units'] for layer in layer_dims])
    height_scale = 500 / max(max_units, 1)  # Scale height, min 1 to avoid div by 0
    
    # Calculate node positions
    x_positions = linspace(0, 1000, len(layer_dims))
    
    # Nodes for each layer
    for i, layer in enumerate(layer_dims):
        units = layer['units']
        y_positions = linspace(0, units * height_scale, min(units, 20))
        
        # If too many nodes, show only a sample
        if units > 20:
            # Add text to indicate more nodes
            fig.add_annotation(
                x=x_positions[i],
                y=y_positions[-1] + 30,
                text=f'+{units - 20} more',
                showarrow=False,
                font=dict(size=10)
            )
        
        # Draw nodes (limit to max 20 visible nodes per layer)
        for j, y in enumerate(y_positions):
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x_positions[i] - 15,
                y0=y - 15,
                x1=x_positions[i] + 15,
                y1=y + 15,
                line_color="LightSkyBlue",
                fillcolor="LightSkyBlue"
            )
        
        # Add layer label
        fig.add_annotation(
            x=x_positions[i],
            y=-50,
            text=f"{layer['type']}<br>({units})",
            showarrow=False,
            font=dict(size=12)
        )
        
        # Draw connections to next layer
        if i < len(layer_dims) - 1:
            next_layer = layer_dims[i + 1]
            next_units = next_layer['units']
            next_y_positions = linspace(0, next_units * height_scale, min(next_units, 20))
            
            # Connect sample nodes (limit to avoid too many lines)
            # Only draw up to 50 connection lines to maintain clarity
            max_connections = min(50, min(len(y_positions), len(next_y_positions)))
            
            # Select nodes for connections
            source_indices = array_linspace(0, len(y_positions) - 1, max_connections, dtype=int)
            target_indices = array_linspace(0, len(next_y_positions) - 1, max_connections, dtype=int)
            
            for j in range(max_connections):
                fig.add_shape(
                    type="line",
                    xref="x", yref="y",
                    x0=x_positions[i] + 15,
                    y0=y_positions[source_indices[j]],
                    x1=x_positions[i + 1] - 15,
                    y1=next_y_positions[target_indices[j]],
                    line=dict(color="Gray", width=0.5)
                )
    
    # Layout
    fig.update_layout(
        title="Network Architecture",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        width=1000,
        margin=dict(l=20, r=20, t=60, b=80)
    )
    
    return fig

def plot_model_summary(model):
    """Generate a text summary of model layers and parameters."""
    if model is None or go is None:
        print("Cannot plot model summary: model or plotly is None")
        return None
        
    # Capture model summary as string
    string_io = io.StringIO()
    model.summary(print_fn=lambda x: string_io.write(x + '\n'))
    summary_string = string_io.getvalue()
    string_io.close()
    
    # Convert to plotly table format
    lines = summary_string.strip().split('\n')
    
    # Extract header and separator lines
    header = lines[2].strip().split()
    
    # Create table data
    table_data = []
    for i in range(3, len(lines) - 4):  # Skip header and final lines
        if '=' not in lines[i]:  # Skip separator lines
            parts = lines[i].strip().split()
            if len(parts) >= 4:
                layer_name = ' '.join(parts[:-3])
                output_shape = parts[-3] if parts[-3] != '(None,' else parts[-3] + ' ' + parts[-2]
                params = parts[-1]
                table_data.append([layer_name, output_shape, params])
    
    # Create table figure
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Layer', 'Output Shape', 'Params'],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=list(zip(*table_data)),
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig.update_layout(title="Model Summary", width=800)
    
    return fig

def visualize_keras_model(model_path):
    """Load and visualize a Keras model from file."""
    try:
        if tf is None:
            print("TensorFlow not available. Cannot load model.")
            return None, None, None
        
        print(f"Loading model from: {model_path}")
        
        # Создаем заглушку для визуализации
        try:
            # Базовые объекты для восстановления моделей
            custom_objects = {}
            
            # Проверяем доступность функций в TensorFlow
            tf_version = tf.__version__
            print(f"TensorFlow version: {tf_version}")
            
            # В TF 2.16+ функции были перемещены
            try:
                from tensorflow.keras.metrics import mean_squared_error
                custom_objects['mse'] = mean_squared_error
                custom_objects['mean_squared_error'] = mean_squared_error
            except (ImportError, AttributeError):
                # Fallback для более новых версий TF
                custom_objects['mse'] = tf.keras.losses.MeanSquaredError()
                custom_objects['mean_squared_error'] = tf.keras.losses.MeanSquaredError()
            
            try:
                from tensorflow.keras.metrics import mean_absolute_error
                custom_objects['mae'] = mean_absolute_error
                custom_objects['mean_absolute_error'] = mean_absolute_error
            except (ImportError, AttributeError):
                custom_objects['mae'] = tf.keras.losses.MeanAbsoluteError()
                custom_objects['mean_absolute_error'] = tf.keras.losses.MeanAbsoluteError()
            
            # Базовые метрики
            custom_objects.update({
                'accuracy': tf.keras.metrics.BinaryAccuracy(),
                'binary_accuracy': tf.keras.metrics.BinaryAccuracy(),
                'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(),
                
                # Потери
                'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
                'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
                
                # Оптимизаторы
                'Adam': tf.keras.optimizers.Adam,
                'SGD': tf.keras.optimizers.SGD,
                'RMSprop': tf.keras.optimizers.RMSprop
            })
            
            # Подклассы для правильной загрузки слоев RNN
            class CustomLayer(tf.keras.layers.Layer):
                def __init__(self, **kwargs):
                    # Удаляем несовместимые аргументы
                    kwargs.pop('time_major', None)
                    super().__init__(**kwargs)
            
            class CustomSimpleRNN(tf.keras.layers.SimpleRNN):
                def __init__(self, *args, **kwargs):
                    # Удаляем несовместимые аргументы
                    kwargs.pop('time_major', None)
                    super().__init__(*args, **kwargs)
            
            class CustomLSTM(tf.keras.layers.LSTM):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('time_major', None)
                    super().__init__(*args, **kwargs)
            
            class CustomGRU(tf.keras.layers.GRU):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('time_major', None)
                    super().__init__(*args, **kwargs)
            
            class CustomBidirectional(tf.keras.layers.Bidirectional):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('time_major', None)
                    super().__init__(*args, **kwargs)
            
            # Добавляем кастомные слои
            custom_objects.update({
                'SimpleRNN': CustomSimpleRNN,
                'LSTM': CustomLSTM,
                'GRU': CustomGRU,
                'Bidirectional': CustomBidirectional,
                'Layer': CustomLayer
            })
            
            # Пробуем различные способы загрузки модели
            model = None
            errors = []
            
            # Метод 1: Без компиляции и с кастомными объектами
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                print("Successfully loaded model with compile=False")
            except Exception as e:
                errors.append(f"Error loading with custom objects, compile=False: {e}")
                
                # Метод 2: Пробуем более простую загрузку
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print("Successfully loaded model with basic load and compile=False")
                except Exception as e:
                    errors.append(f"Error with basic load and compile=False: {e}")
                    
                    # Метод 3: Последняя попытка - создаем мок-модель для визуализации
                    try:
                        # Создаем простую модель-заглушку для отображения
                        inputs = tf.keras.layers.Input(shape=(10,))
                        x = tf.keras.layers.Dense(5, activation='relu')(inputs)
                        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                        model = tf.keras.Model(inputs=inputs, outputs=outputs)
                        model._name = f"Mock model for {os.path.basename(model_path)}"
                        print(f"Created mock model as fallback")
                    except Exception as e:
                        errors.append(f"Error creating mock model: {e}")
                        return None, None, None
            
            if model is None:
                print(f"Failed to load model with all methods. Errors: {errors}")
                return None, None, None
            
            # Создаем визуализации
            try:
                architecture_fig = plot_model_architecture(model)
                print("Successfully created architecture visualization")
            except Exception as e:
                print(f"Error creating architecture visualization: {e}")
                architecture_fig = None
            
            # Создаем сводку модели
            try:
                summary_fig = plot_model_summary(model)
                print("Successfully created model summary")
            except Exception as e:
                print(f"Error creating model summary: {e}")
                summary_fig = None
            
            return model, architecture_fig, summary_fig
        except Exception as e:
            print(f"Error preparing model visualization: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    except Exception as e:
        print(f"Fatal error in visualize_keras_model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None 