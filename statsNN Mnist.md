```python
import numpy as np
import mnist
from tensorflow import keras


```


```python
# The first time you run this might be a bit slow, since the
# mnist package has to download and cache the data.
train_images = mnist.train_images()
train_labels = mnist.train_labels()


print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
```

    (60000, 28, 28)
    (60000,)
    


```python
import numpy as np
import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()


# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape) # (60000, 784)
print(test_images.shape)  # (10000, 784)
```

    (60000, 784)
    (10000, 784)
    


```python
input_shape = train_images.shape[1]
input_shape
```




    784




```python

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
```


```python

model=Sequential()
model.add(Dense(units=128, input_dim=input_shape, activation='relu'))
model.add(Dense(units=256, input_dim=input_shape, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_5 (Dense)              (None, 128)               100480    
    _________________________________________________________________
    dense_6 (Dense)              (None, 256)               33024     
    _________________________________________________________________
    dense_7 (Dense)              (None, 512)               131584    
    _________________________________________________________________
    dense_8 (Dense)              (None, 64)                32832     
    _________________________________________________________________
    dense_9 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 298,570
    Trainable params: 298,570
    Non-trainable params: 0
    _________________________________________________________________
    


```python
history=model.fit(train_images,to_categorical(train_labels), epochs=50,  verbose=1, validation_split = 0.1, shuffle=False)

```

    Train on 54000 samples, validate on 6000 samples
    Epoch 1/50
       32/54000 [..............................] - ETA: 2:53


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\framework\ops.py in _create_c_op(graph, node_def, inputs, control_inputs)
       1618   try:
    -> 1619     c_op = c_api.TF_FinishOperation(op_desc)
       1620   except errors.InvalidArgumentError as e:
    

    InvalidArgumentError: Can not squeeze dim[1], expected a dimension of 1, got 10 for 'metrics/accuracy/Squeeze' (op: 'Squeeze') with input shapes: [?,10].

    
    During handling of the above exception, another exception occurred:
    

    ValueError                                Traceback (most recent call last)

    <ipython-input-11-dea03f22c756> in <module>
    ----> 1 history=model.fit(train_images,to_categorical(train_labels), epochs=50,  verbose=1, validation_split = 0.1, shuffle=False)
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        817         max_queue_size=max_queue_size,
        818         workers=workers,
    --> 819         use_multiprocessing=use_multiprocessing)
        820 
        821   def evaluate(self,
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        340                 mode=ModeKeys.TRAIN,
        341                 training_context=training_context,
    --> 342                 total_epochs=epochs)
        343             cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)
        344 
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_v2.py in run_one_epoch(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)
        126         step=step, mode=mode, size=current_batch_size) as batch_logs:
        127       try:
    --> 128         batch_outs = execution_function(iterator)
        129       except (StopIteration, errors.OutOfRangeError):
        130         # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_v2_utils.py in execution_function(input_fn)
         96     # `numpy` translates Tensors to values in Eager mode.
         97     return nest.map_structure(_non_none_constant_value,
    ---> 98                               distributed_function(input_fn))
         99 
        100   return execution_function
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\eager\def_function.py in __call__(self, *args, **kwds)
        566         xla_context.Exit()
        567     else:
    --> 568       result = self._call(*args, **kwds)
        569 
        570     if tracing_count == self._get_tracing_count():
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\eager\def_function.py in _call(self, *args, **kwds)
        613       # This is the first call of __call__, so we have to initialize.
        614       initializers = []
    --> 615       self._initialize(args, kwds, add_initializers_to=initializers)
        616     finally:
        617       # At this point we know that the initialization is complete (or less
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\eager\def_function.py in _initialize(self, args, kwds, add_initializers_to)
        495     self._concrete_stateful_fn = (
        496         self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
    --> 497             *args, **kwds))
        498 
        499     def invalid_creator_scope(*unused_args, **unused_kwds):
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\eager\function.py in _get_concrete_function_internal_garbage_collected(self, *args, **kwargs)
       2387       args, kwargs = None, None
       2388     with self._lock:
    -> 2389       graph_function, _, _ = self._maybe_define_function(args, kwargs)
       2390     return graph_function
       2391 
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\eager\function.py in _maybe_define_function(self, args, kwargs)
       2701 
       2702       self._function_cache.missed.add(call_context_key)
    -> 2703       graph_function = self._create_graph_function(args, kwargs)
       2704       self._function_cache.primary[cache_key] = graph_function
       2705       return graph_function, args, kwargs
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\eager\function.py in _create_graph_function(self, args, kwargs, override_flat_arg_shapes)
       2591             arg_names=arg_names,
       2592             override_flat_arg_shapes=override_flat_arg_shapes,
    -> 2593             capture_by_value=self._capture_by_value),
       2594         self._function_attributes,
       2595         # Tell the ConcreteFunction to clean up its graph once it goes out of
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\framework\func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, autograph, autograph_options, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, override_flat_arg_shapes)
        976                                           converted_func)
        977 
    --> 978       func_outputs = python_func(*func_args, **func_kwargs)
        979 
        980       # invariant: `func_outputs` contains only Tensors, CompositeTensors,
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\eager\def_function.py in wrapped_fn(*args, **kwds)
        437         # __wrapped__ allows AutoGraph to swap in a converted function. We give
        438         # the function a weak reference to itself to avoid a reference cycle.
    --> 439         return weak_wrapped_fn().__wrapped__(*args, **kwds)
        440     weak_wrapped_fn = weakref.ref(wrapped_fn)
        441 
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_v2_utils.py in distributed_function(input_iterator)
         83     args = _prepare_feed_values(model, input_iterator, mode, strategy)
         84     outputs = strategy.experimental_run_v2(
    ---> 85         per_replica_function, args=args)
         86     # Out of PerReplica outputs reduce or pick values to return.
         87     all_outputs = dist_utils.unwrap_output_dict(
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\distribute\distribute_lib.py in experimental_run_v2(self, fn, args, kwargs)
        761       fn = autograph.tf_convert(fn, ag_ctx.control_status_ctx(),
        762                                 convert_by_default=False)
    --> 763       return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
        764 
        765   def reduce(self, reduce_op, value, axis):
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\distribute\distribute_lib.py in call_for_each_replica(self, fn, args, kwargs)
       1817       kwargs = {}
       1818     with self._container_strategy().scope():
    -> 1819       return self._call_for_each_replica(fn, args, kwargs)
       1820 
       1821   def _call_for_each_replica(self, fn, args, kwargs):
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\distribute\distribute_lib.py in _call_for_each_replica(self, fn, args, kwargs)
       2162         self._container_strategy(),
       2163         replica_id_in_sync_group=constant_op.constant(0, dtypes.int32)):
    -> 2164       return fn(*args, **kwargs)
       2165 
       2166   def _reduce_to(self, reduce_op, value, destinations):
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\autograph\impl\api.py in wrapper(*args, **kwargs)
        290   def wrapper(*args, **kwargs):
        291     with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED):
    --> 292       return func(*args, **kwargs)
        293 
        294   if inspect.isfunction(func) or inspect.ismethod(func):
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_v2_utils.py in train_on_batch(model, x, y, sample_weight, class_weight, reset_metrics, standalone)
        431       y,
        432       sample_weights=sample_weights,
    --> 433       output_loss_metrics=model._output_loss_metrics)
        434 
        435   if reset_metrics:
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_eager.py in train_on_batch(model, inputs, targets, sample_weights, output_loss_metrics)
        314     outs = [outs]
        315   metrics_results = _eager_metrics_fn(
    --> 316       model, outs, targets, sample_weights=sample_weights, masks=masks)
        317   total_loss = nest.flatten(total_loss)
        318   return {'total_loss': total_loss,
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_eager.py in _eager_metrics_fn(model, outputs, targets, sample_weights, masks)
         72         masks=masks,
         73         return_weighted_and_unweighted_metrics=True,
    ---> 74         skip_target_masks=model._prepare_skip_target_masks())
         75 
         76   # Add metric results from the `add_metric` metrics.
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training.py in _handle_metrics(self, outputs, targets, skip_target_masks, sample_weights, masks, return_weighted_metrics, return_weighted_and_unweighted_metrics)
       2002           metric_results.extend(
       2003               self._handle_per_output_metrics(self._per_output_metrics[i],
    -> 2004                                               target, output, output_mask))
       2005         if return_weighted_and_unweighted_metrics or return_weighted_metrics:
       2006           metric_results.extend(
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training.py in _handle_per_output_metrics(self, metrics_dict, y_true, y_pred, mask, weights)
       1953       with K.name_scope(metric_name):
       1954         metric_result = training_utils.call_metric_function(
    -> 1955             metric_fn, y_true, y_pred, weights=weights, mask=mask)
       1956         metric_results.append(metric_result)
       1957     return metric_results
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\engine\training_utils.py in call_metric_function(metric_fn, y_true, y_pred, weights, mask)
       1153 
       1154   if y_pred is not None:
    -> 1155     return metric_fn(y_true, y_pred, sample_weight=weights)
       1156   # `Mean` metric only takes a single value.
       1157   return metric_fn(y_true, sample_weight=weights)
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\metrics.py in __call__(self, *args, **kwargs)
        194     from tensorflow.python.keras.distribute import distributed_training_utils  # pylint:disable=g-import-not-at-top
        195     return distributed_training_utils.call_replica_local_fn(
    --> 196         replica_local_fn, *args, **kwargs)
        197 
        198   @property
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\distribute\distributed_training_utils.py in call_replica_local_fn(fn, *args, **kwargs)
       1133     with strategy.scope():
       1134       return strategy.extended.call_for_each_replica(fn, args, kwargs)
    -> 1135   return fn(*args, **kwargs)
       1136 
       1137 
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\metrics.py in replica_local_fn(*args, **kwargs)
        177     def replica_local_fn(*args, **kwargs):
        178       """Updates the state of the metric in a replica-local context."""
    --> 179       update_op = self.update_state(*args, **kwargs)  # pylint: disable=not-callable
        180       with ops.control_dependencies([update_op]):
        181         result_t = self.result()  # pylint: disable=not-callable
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\utils\metrics_utils.py in decorated(metric_obj, *args, **kwargs)
         74 
         75     with tf_utils.graph_context_for_symbolic_tensors(*args, **kwargs):
    ---> 76       update_op = update_state_fn(*args, **kwargs)
         77     if update_op is not None:  # update_op will be None in eager execution.
         78       metric_obj.add_update(update_op)
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\metrics.py in update_state(self, y_true, y_pred, sample_weight)
        585         y_pred, y_true)
        586 
    --> 587     matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        588     return super(MeanMetricWrapper, self).update_state(
        589         matches, sample_weight=sample_weight)
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\keras\metrics.py in sparse_categorical_accuracy(y_true, y_pred)
       2979   if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
       2980       K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    -> 2981     y_true = array_ops.squeeze(y_true, [-1])
       2982   y_pred = math_ops.argmax(y_pred, axis=-1)
       2983 
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\util\dispatch.py in wrapper(*args, **kwargs)
        178     """Call target, and fall back on dispatchers if there is a TypeError."""
        179     try:
    --> 180       return target(*args, **kwargs)
        181     except (TypeError, ValueError):
        182       # Note: convert_to_eager_tensor currently raises a ValueError, not a
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\util\deprecation.py in new_func(*args, **kwargs)
        505                 'in a future version' if date is None else ('after %s' % date),
        506                 instructions)
    --> 507       return func(*args, **kwargs)
        508 
        509     doc = _add_deprecated_arg_notice_to_docstring(
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\ops\array_ops.py in squeeze(input, axis, name, squeeze_dims)
       3778   if np.isscalar(axis):
       3779     axis = [axis]
    -> 3780   return gen_array_ops.squeeze(input, axis, name)
       3781 
       3782 
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\ops\gen_array_ops.py in squeeze(input, axis, name)
       9229   axis = [_execute.make_int(_i, "axis") for _i in axis]
       9230   _, _, _op, _outputs = _op_def_library._apply_op_helper(
    -> 9231         "Squeeze", input=input, squeeze_dims=axis, name=name)
       9232   _result = _outputs[:]
       9233   if _execute.must_record_gradient():
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\framework\op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
        740       op = g._create_op_internal(op_type_name, inputs, dtypes=None,
        741                                  name=scope, input_types=input_types,
    --> 742                                  attrs=attr_protos, op_def=op_def)
        743 
        744     # `outputs` is returned as a separate return value so that the output
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\framework\func_graph.py in _create_op_internal(self, op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
        593     return super(FuncGraph, self)._create_op_internal(  # pylint: disable=protected-access
        594         op_type, inputs, dtypes, input_types, name, attrs, op_def,
    --> 595         compute_device)
        596 
        597   def capture(self, tensor, name=None, shape=None):
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\framework\ops.py in _create_op_internal(self, op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
       3320           input_types=input_types,
       3321           original_op=self._default_original_op,
    -> 3322           op_def=op_def)
       3323       self._create_op_helper(ret, compute_device=compute_device)
       3324     return ret
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\framework\ops.py in __init__(self, node_def, g, inputs, output_types, control_inputs, input_types, original_op, op_def)
       1784           op_def, inputs, node_def.attr)
       1785       self._c_op = _create_c_op(self._graph, node_def, grouped_inputs,
    -> 1786                                 control_input_ops)
       1787       name = compat.as_str(node_def.name)
       1788     # pylint: enable=protected-access
    

    c:\users\s4562394\.conda\envs\tf2\lib\site-packages\tensorflow_core\python\framework\ops.py in _create_c_op(graph, node_def, inputs, control_inputs)
       1620   except errors.InvalidArgumentError as e:
       1621     # Convert to ValueError for backwards compatibility.
    -> 1622     raise ValueError(str(e))
       1623 
       1624   return c_op
    

    ValueError: Can not squeeze dim[1], expected a dimension of 1, got 10 for 'metrics/accuracy/Squeeze' (op: 'Squeeze') with input shapes: [?,10].



```python
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_images,  to_categorical(test_labels), batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_images[:3])
print("predictions shape:", predictions.shape)
```


```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')


plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();
```


```python
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')


plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();
```


```python
def tf_plot_gallery(images, h, w,titles = [0], n_row=3, n_col=4):
    """Helper function to plot a gallery of tensor portraits"""
    
    import tensorflow as tf
    if len(titles) == 0:
        titles = np.arange(len(images))
    
    
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(tf.reshape(images[i],[h, w]), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
```


```python
tf_plot_gallery(train_images, 28,28, train_labels, 5,5)
```


```python
y_predict = model.predict(test_images[:25])
```


```python

y_predct = np.argmax(y_predict, axis=-1)
print(y_predct)
# [0, 1, 2, 0, 4, 5]
```


```python
tf_plot_gallery(test_images[:25],28,28,y_predct,5,5)
```


```python
tf_plot_gallery(test_images[:25],28,28,test_labels[:25],5,5)
```
