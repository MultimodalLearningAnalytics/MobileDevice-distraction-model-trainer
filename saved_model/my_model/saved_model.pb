??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?1d* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	?1d*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:d*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:d*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv_lst_m2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameconv_lst_m2d_9/kernel
?
)conv_lst_m2d_9/kernel/Read/ReadVariableOpReadVariableOpconv_lst_m2d_9/kernel*'
_output_shapes
:	?*
dtype0
?
conv_lst_m2d_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*0
shared_name!conv_lst_m2d_9/recurrent_kernel
?
3conv_lst_m2d_9/recurrent_kernel/Read/ReadVariableOpReadVariableOpconv_lst_m2d_9/recurrent_kernel*'
_output_shapes
:@?*
dtype0

conv_lst_m2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameconv_lst_m2d_9/bias
x
'conv_lst_m2d_9/bias/Read/ReadVariableOpReadVariableOpconv_lst_m2d_9/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?1d*'
shared_nameAdam/dense_18/kernel/m
?
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes
:	?1d*
dtype0
?
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_19/kernel/m
?
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv_lst_m2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_nameAdam/conv_lst_m2d_9/kernel/m
?
0Adam/conv_lst_m2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_9/kernel/m*'
_output_shapes
:	?*
dtype0
?
&Adam/conv_lst_m2d_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*7
shared_name(&Adam/conv_lst_m2d_9/recurrent_kernel/m
?
:Adam/conv_lst_m2d_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/conv_lst_m2d_9/recurrent_kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv_lst_m2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/conv_lst_m2d_9/bias/m
?
.Adam/conv_lst_m2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_9/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?1d*'
shared_nameAdam/dense_18/kernel/v
?
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes
:	?1d*
dtype0
?
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_19/kernel/v
?
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv_lst_m2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_nameAdam/conv_lst_m2d_9/kernel/v
?
0Adam/conv_lst_m2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_9/kernel/v*'
_output_shapes
:	?*
dtype0
?
&Adam/conv_lst_m2d_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*7
shared_name(&Adam/conv_lst_m2d_9/recurrent_kernel/v
?
:Adam/conv_lst_m2d_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/conv_lst_m2d_9/recurrent_kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv_lst_m2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/conv_lst_m2d_9/bias/v
?
.Adam/conv_lst_m2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_lst_m2d_9/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?/
value?.B?. B?.
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
?
&iter

'beta_1

(beta_2
	)decay
*learning_ratemamb mc!md+me,mf-mgvhvi vj!vk+vl,vm-vn
1
+0
,1
-2
3
4
 5
!6
 
1
+0
,1
-2
3
4
 5
!6
?

.layers
/layer_metrics
0metrics
	variables
1layer_regularization_losses
regularization_losses
2non_trainable_variables
	trainable_variables
 
~

+kernel
,recurrent_kernel
-bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
 

+0
,1
-2
 

+0
,1
-2
?

7layers
8layer_metrics
9metrics
	variables

:states
;layer_regularization_losses
regularization_losses
<non_trainable_variables
trainable_variables
 
 
 
?

=layers
>layer_metrics
?metrics
	variables
@layer_regularization_losses
regularization_losses
Anon_trainable_variables
trainable_variables
 
 
 
?

Blayers
Clayer_metrics
Dmetrics
	variables
Elayer_regularization_losses
regularization_losses
Fnon_trainable_variables
trainable_variables
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Glayers
Hlayer_metrics
Imetrics
	variables
Jlayer_regularization_losses
regularization_losses
Knon_trainable_variables
trainable_variables
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?

Llayers
Mlayer_metrics
Nmetrics
"	variables
Olayer_regularization_losses
#regularization_losses
Pnon_trainable_variables
$trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv_lst_m2d_9/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv_lst_m2d_9/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEconv_lst_m2d_9/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4
 

Q0
R1
 
 

+0
,1
-2
 

+0
,1
-2
?

Slayers
Tlayer_metrics
Umetrics
3	variables
Vlayer_regularization_losses
4regularization_losses
Wnon_trainable_variables
5trainable_variables

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Xtotal
	Ycount
Z	variables
[	keras_api
D
	\total
	]count
^
_fn_kwargs
_	variables
`	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

Z	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

\0
]1

_	variables
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv_lst_m2d_9/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/conv_lst_m2d_9/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/conv_lst_m2d_9/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv_lst_m2d_9/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/conv_lst_m2d_9/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/conv_lst_m2d_9/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
$serving_default_conv_lst_m2d_9_inputPlaceholder*3
_output_shapes!
:?????????d	*
dtype0*(
shape:?????????d	
?
StatefulPartitionedCallStatefulPartitionedCall$serving_default_conv_lst_m2d_9_inputconv_lst_m2d_9/kernelconv_lst_m2d_9/recurrent_kernelconv_lst_m2d_9/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_62009
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)conv_lst_m2d_9/kernel/Read/ReadVariableOp3conv_lst_m2d_9/recurrent_kernel/Read/ReadVariableOp'conv_lst_m2d_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp0Adam/conv_lst_m2d_9/kernel/m/Read/ReadVariableOp:Adam/conv_lst_m2d_9/recurrent_kernel/m/Read/ReadVariableOp.Adam/conv_lst_m2d_9/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp0Adam/conv_lst_m2d_9/kernel/v/Read/ReadVariableOp:Adam/conv_lst_m2d_9/recurrent_kernel/v/Read/ReadVariableOp.Adam/conv_lst_m2d_9/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_63839
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/biasdense_19/kerneldense_19/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv_lst_m2d_9/kernelconv_lst_m2d_9/recurrent_kernelconv_lst_m2d_9/biastotalcounttotal_1count_1Adam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/conv_lst_m2d_9/kernel/m&Adam/conv_lst_m2d_9/recurrent_kernel/mAdam/conv_lst_m2d_9/bias/mAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/vAdam/conv_lst_m2d_9/kernel/v&Adam/conv_lst_m2d_9/recurrent_kernel/vAdam/conv_lst_m2d_9/bias/v**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_63939??
??
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_62532

inputsG
,conv_lst_m2d_9_split_readvariableop_resource:	?I
.conv_lst_m2d_9_split_1_readvariableop_resource:@?=
.conv_lst_m2d_9_split_2_readvariableop_resource:	?:
'dense_18_matmul_readvariableop_resource:	?1d6
(dense_18_biasadd_readvariableop_resource:d9
'dense_19_matmul_readvariableop_resource:d6
(dense_19_biasadd_readvariableop_resource:
identity??#conv_lst_m2d_9/split/ReadVariableOp?%conv_lst_m2d_9/split_1/ReadVariableOp?%conv_lst_m2d_9/split_2/ReadVariableOp?conv_lst_m2d_9/while?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
conv_lst_m2d_9/zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????d	2
conv_lst_m2d_9/zeros_like?
$conv_lst_m2d_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_9/Sum/reduction_indices?
conv_lst_m2d_9/SumSumconv_lst_m2d_9/zeros_like:y:0-conv_lst_m2d_9/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
conv_lst_m2d_9/Sum?
$conv_lst_m2d_9/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2&
$conv_lst_m2d_9/zeros/shape_as_tensor}
conv_lst_m2d_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv_lst_m2d_9/zeros/Const?
conv_lst_m2d_9/zerosFill-conv_lst_m2d_9/zeros/shape_as_tensor:output:0#conv_lst_m2d_9/zeros/Const:output:0*
T0*&
_output_shapes
:	@2
conv_lst_m2d_9/zeros?
conv_lst_m2d_9/convolutionConv2Dconv_lst_m2d_9/Sum:output:0conv_lst_m2d_9/zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution?
conv_lst_m2d_9/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d_9/transpose/perm?
conv_lst_m2d_9/transpose	Transposeinputs&conv_lst_m2d_9/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????d	2
conv_lst_m2d_9/transposex
conv_lst_m2d_9/ShapeShapeconv_lst_m2d_9/transpose:y:0*
T0*
_output_shapes
:2
conv_lst_m2d_9/Shape?
"conv_lst_m2d_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"conv_lst_m2d_9/strided_slice/stack?
$conv_lst_m2d_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_9/strided_slice/stack_1?
$conv_lst_m2d_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_9/strided_slice/stack_2?
conv_lst_m2d_9/strided_sliceStridedSliceconv_lst_m2d_9/Shape:output:0+conv_lst_m2d_9/strided_slice/stack:output:0-conv_lst_m2d_9/strided_slice/stack_1:output:0-conv_lst_m2d_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
conv_lst_m2d_9/strided_slice?
*conv_lst_m2d_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*conv_lst_m2d_9/TensorArrayV2/element_shape?
conv_lst_m2d_9/TensorArrayV2TensorListReserve3conv_lst_m2d_9/TensorArrayV2/element_shape:output:0%conv_lst_m2d_9/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d_9/TensorArrayV2?
Dconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   2F
Dconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
6conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lst_m2d_9/transpose:y:0Mconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor?
$conv_lst_m2d_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_lst_m2d_9/strided_slice_1/stack?
&conv_lst_m2d_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_9/strided_slice_1/stack_1?
&conv_lst_m2d_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_9/strided_slice_1/stack_2?
conv_lst_m2d_9/strided_slice_1StridedSliceconv_lst_m2d_9/transpose:y:0-conv_lst_m2d_9/strided_slice_1/stack:output:0/conv_lst_m2d_9/strided_slice_1/stack_1:output:0/conv_lst_m2d_9/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2 
conv_lst_m2d_9/strided_slice_1?
conv_lst_m2d_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv_lst_m2d_9/split/split_dim?
#conv_lst_m2d_9/split/ReadVariableOpReadVariableOp,conv_lst_m2d_9_split_readvariableop_resource*'
_output_shapes
:	?*
dtype02%
#conv_lst_m2d_9/split/ReadVariableOp?
conv_lst_m2d_9/splitSplit'conv_lst_m2d_9/split/split_dim:output:0+conv_lst_m2d_9/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
conv_lst_m2d_9/split?
 conv_lst_m2d_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 conv_lst_m2d_9/split_1/split_dim?
%conv_lst_m2d_9/split_1/ReadVariableOpReadVariableOp.conv_lst_m2d_9_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02'
%conv_lst_m2d_9/split_1/ReadVariableOp?
conv_lst_m2d_9/split_1Split)conv_lst_m2d_9/split_1/split_dim:output:0-conv_lst_m2d_9/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_9/split_1?
 conv_lst_m2d_9/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv_lst_m2d_9/split_2/split_dim?
%conv_lst_m2d_9/split_2/ReadVariableOpReadVariableOp.conv_lst_m2d_9_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%conv_lst_m2d_9/split_2/ReadVariableOp?
conv_lst_m2d_9/split_2Split)conv_lst_m2d_9/split_2/split_dim:output:0-conv_lst_m2d_9/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_9/split_2?
conv_lst_m2d_9/convolution_1Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_1?
conv_lst_m2d_9/BiasAddBiasAdd%conv_lst_m2d_9/convolution_1:output:0conv_lst_m2d_9/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd?
conv_lst_m2d_9/convolution_2Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_2?
conv_lst_m2d_9/BiasAdd_1BiasAdd%conv_lst_m2d_9/convolution_2:output:0conv_lst_m2d_9/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd_1?
conv_lst_m2d_9/convolution_3Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_3?
conv_lst_m2d_9/BiasAdd_2BiasAdd%conv_lst_m2d_9/convolution_3:output:0conv_lst_m2d_9/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd_2?
conv_lst_m2d_9/convolution_4Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_4?
conv_lst_m2d_9/BiasAdd_3BiasAdd%conv_lst_m2d_9/convolution_4:output:0conv_lst_m2d_9/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd_3?
conv_lst_m2d_9/convolution_5Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_5?
conv_lst_m2d_9/convolution_6Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_6?
conv_lst_m2d_9/convolution_7Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_7?
conv_lst_m2d_9/convolution_8Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_8?
conv_lst_m2d_9/addAddV2conv_lst_m2d_9/BiasAdd:output:0%conv_lst_m2d_9/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/addq
conv_lst_m2d_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/Constu
conv_lst_m2d_9/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/Const_1?
conv_lst_m2d_9/MulMulconv_lst_m2d_9/add:z:0conv_lst_m2d_9/Const:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Mul?
conv_lst_m2d_9/Add_1Addconv_lst_m2d_9/Mul:z:0conv_lst_m2d_9/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Add_1?
&conv_lst_m2d_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d_9/clip_by_value/Minimum/y?
$conv_lst_m2d_9/clip_by_value/MinimumMinimumconv_lst_m2d_9/Add_1:z:0/conv_lst_m2d_9/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2&
$conv_lst_m2d_9/clip_by_value/Minimum?
conv_lst_m2d_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d_9/clip_by_value/y?
conv_lst_m2d_9/clip_by_valueMaximum(conv_lst_m2d_9/clip_by_value/Minimum:z:0'conv_lst_m2d_9/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/clip_by_value?
conv_lst_m2d_9/add_2AddV2!conv_lst_m2d_9/BiasAdd_1:output:0%conv_lst_m2d_9/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_2u
conv_lst_m2d_9/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/Const_2u
conv_lst_m2d_9/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/Const_3?
conv_lst_m2d_9/Mul_1Mulconv_lst_m2d_9/add_2:z:0conv_lst_m2d_9/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Mul_1?
conv_lst_m2d_9/Add_3Addconv_lst_m2d_9/Mul_1:z:0conv_lst_m2d_9/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Add_3?
(conv_lst_m2d_9/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_9/clip_by_value_1/Minimum/y?
&conv_lst_m2d_9/clip_by_value_1/MinimumMinimumconv_lst_m2d_9/Add_3:z:01conv_lst_m2d_9/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2(
&conv_lst_m2d_9/clip_by_value_1/Minimum?
 conv_lst_m2d_9/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_9/clip_by_value_1/y?
conv_lst_m2d_9/clip_by_value_1Maximum*conv_lst_m2d_9/clip_by_value_1/Minimum:z:0)conv_lst_m2d_9/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/clip_by_value_1?
conv_lst_m2d_9/mul_2Mul"conv_lst_m2d_9/clip_by_value_1:z:0#conv_lst_m2d_9/convolution:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/mul_2?
conv_lst_m2d_9/add_4AddV2!conv_lst_m2d_9/BiasAdd_2:output:0%conv_lst_m2d_9/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_4?
conv_lst_m2d_9/ReluReluconv_lst_m2d_9/add_4:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Relu?
conv_lst_m2d_9/mul_3Mul conv_lst_m2d_9/clip_by_value:z:0!conv_lst_m2d_9/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/mul_3?
conv_lst_m2d_9/add_5AddV2conv_lst_m2d_9/mul_2:z:0conv_lst_m2d_9/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_5?
conv_lst_m2d_9/add_6AddV2!conv_lst_m2d_9/BiasAdd_3:output:0%conv_lst_m2d_9/convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_6u
conv_lst_m2d_9/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/Const_4u
conv_lst_m2d_9/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/Const_5?
conv_lst_m2d_9/Mul_4Mulconv_lst_m2d_9/add_6:z:0conv_lst_m2d_9/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Mul_4?
conv_lst_m2d_9/Add_7Addconv_lst_m2d_9/Mul_4:z:0conv_lst_m2d_9/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Add_7?
(conv_lst_m2d_9/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_9/clip_by_value_2/Minimum/y?
&conv_lst_m2d_9/clip_by_value_2/MinimumMinimumconv_lst_m2d_9/Add_7:z:01conv_lst_m2d_9/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2(
&conv_lst_m2d_9/clip_by_value_2/Minimum?
 conv_lst_m2d_9/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_9/clip_by_value_2/y?
conv_lst_m2d_9/clip_by_value_2Maximum*conv_lst_m2d_9/clip_by_value_2/Minimum:z:0)conv_lst_m2d_9/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/clip_by_value_2?
conv_lst_m2d_9/Relu_1Reluconv_lst_m2d_9/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Relu_1?
conv_lst_m2d_9/mul_5Mul"conv_lst_m2d_9/clip_by_value_2:z:0#conv_lst_m2d_9/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/mul_5?
,conv_lst_m2d_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2.
,conv_lst_m2d_9/TensorArrayV2_1/element_shape?
conv_lst_m2d_9/TensorArrayV2_1TensorListReserve5conv_lst_m2d_9/TensorArrayV2_1/element_shape:output:0%conv_lst_m2d_9/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
conv_lst_m2d_9/TensorArrayV2_1l
conv_lst_m2d_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
conv_lst_m2d_9/time?
'conv_lst_m2d_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'conv_lst_m2d_9/while/maximum_iterations?
!conv_lst_m2d_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv_lst_m2d_9/while/loop_counter?
conv_lst_m2d_9/whileWhile*conv_lst_m2d_9/while/loop_counter:output:00conv_lst_m2d_9/while/maximum_iterations:output:0conv_lst_m2d_9/time:output:0'conv_lst_m2d_9/TensorArrayV2_1:handle:0#conv_lst_m2d_9/convolution:output:0#conv_lst_m2d_9/convolution:output:0%conv_lst_m2d_9/strided_slice:output:0Fconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0,conv_lst_m2d_9_split_readvariableop_resource.conv_lst_m2d_9_split_1_readvariableop_resource.conv_lst_m2d_9_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
conv_lst_m2d_9_while_body_62382*+
cond#R!
conv_lst_m2d_9_while_cond_62381*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
conv_lst_m2d_9/while?
?conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2A
?conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shape?
1conv_lst_m2d_9/TensorArrayV2Stack/TensorListStackTensorListStackconv_lst_m2d_9/while:output:3Hconv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????b@*
element_dtype023
1conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack?
$conv_lst_m2d_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$conv_lst_m2d_9/strided_slice_2/stack?
&conv_lst_m2d_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_lst_m2d_9/strided_slice_2/stack_1?
&conv_lst_m2d_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_9/strided_slice_2/stack_2?
conv_lst_m2d_9/strided_slice_2StridedSlice:conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack:tensor:0-conv_lst_m2d_9/strided_slice_2/stack:output:0/conv_lst_m2d_9/strided_slice_2/stack_1:output:0/conv_lst_m2d_9/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2 
conv_lst_m2d_9/strided_slice_2?
conv_lst_m2d_9/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2!
conv_lst_m2d_9/transpose_1/perm?
conv_lst_m2d_9/transpose_1	Transpose:conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack:tensor:0(conv_lst_m2d_9/transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????b@2
conv_lst_m2d_9/transpose_1w
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul'conv_lst_m2d_9/strided_slice_2:output:0 dropout_9/dropout/Const:output:0*
T0*/
_output_shapes
:?????????b@2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape'conv_lst_m2d_9/strided_slice_2:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????b@*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????b@2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????b@2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????b@2
dropout_9/dropout/Mul_1s
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_9/Const?
flatten_9/ReshapeReshapedropout_9/dropout/Mul_1:z:0flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????12
flatten_9/Reshape?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	?1d*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMulflatten_9/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_18/Relu?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Softmax?
IdentityIdentitydense_19/Softmax:softmax:0$^conv_lst_m2d_9/split/ReadVariableOp&^conv_lst_m2d_9/split_1/ReadVariableOp&^conv_lst_m2d_9/split_2/ReadVariableOp^conv_lst_m2d_9/while ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 2J
#conv_lst_m2d_9/split/ReadVariableOp#conv_lst_m2d_9/split/ReadVariableOp2N
%conv_lst_m2d_9/split_1/ReadVariableOp%conv_lst_m2d_9/split_1/ReadVariableOp2N
%conv_lst_m2d_9/split_2/ReadVariableOp%conv_lst_m2d_9/split_2/ReadVariableOp2,
conv_lst_m2d_9/whileconv_lst_m2d_9/while2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?8
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_61028

inputs"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?whilet

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&??????????????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????b@:?????????b@:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_608962
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_60960*
condR
while_cond_60959*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^StatefulPartitionedCall^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&??????????????????d	: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&??????????????????d	
 
_user_specified_nameinputs
?

?
C__inference_dense_18_layer_call_and_return_conditional_losses_63522

inputs1
matmul_readvariableop_resource:	?1d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?1d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????1
 
_user_specified_nameinputs
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_61546

inputs/
conv_lst_m2d_9_61489:	?/
conv_lst_m2d_9_61491:@?#
conv_lst_m2d_9_61493:	?!
dense_18_61523:	?1d
dense_18_61525:d 
dense_19_61540:d
dense_19_61542:
identity??&conv_lst_m2d_9/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
&conv_lst_m2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv_lst_m2d_9_61489conv_lst_m2d_9_61491conv_lst_m2d_9_61493*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_614882(
&conv_lst_m2d_9/StatefulPartitionedCall?
dropout_9/PartitionedCallPartitionedCall/conv_lst_m2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_615012
dropout_9/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_615092
flatten_9/PartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_18_61523dense_18_61525*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_615222"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_61540dense_19_61542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_615392"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0'^conv_lst_m2d_9/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 2P
&conv_lst_m2d_9/StatefulPartitionedCall&conv_lst_m2d_9/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
?
while_cond_63115
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_63115___redundant_placeholder03
/while_while_cond_63115___redundant_placeholder13
/while_while_cond_63115___redundant_placeholder23
/while_while_cond_63115___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?	
?
,__inference_sequential_9_layer_call_fn_61563
conv_lst_m2d_9_input"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?1d
	unknown_3:d
	unknown_4:d
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv_lst_m2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_615462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
3
_output_shapes!
:?????????d	
.
_user_specified_nameconv_lst_m2d_9_input
?
E
)__inference_dropout_9_layer_call_fn_63469

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_615012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
?E
?
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_63726

inputs
states_0
states_18
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7w
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
IdentityIdentity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity?

Identity_1Identity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_1?

Identity_2Identity	add_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????d	:?????????b@:?????????b@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:?????????d	
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/1
?
?
.__inference_conv_lst_m2d_9_layer_call_fn_62565

inputs"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_614882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????d	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
b
)__inference_dropout_9_layer_call_fn_63474

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_616092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_60604
conv_lst_m2d_9_inputT
9sequential_9_conv_lst_m2d_9_split_readvariableop_resource:	?V
;sequential_9_conv_lst_m2d_9_split_1_readvariableop_resource:@?J
;sequential_9_conv_lst_m2d_9_split_2_readvariableop_resource:	?G
4sequential_9_dense_18_matmul_readvariableop_resource:	?1dC
5sequential_9_dense_18_biasadd_readvariableop_resource:dF
4sequential_9_dense_19_matmul_readvariableop_resource:dC
5sequential_9_dense_19_biasadd_readvariableop_resource:
identity??0sequential_9/conv_lst_m2d_9/split/ReadVariableOp?2sequential_9/conv_lst_m2d_9/split_1/ReadVariableOp?2sequential_9/conv_lst_m2d_9/split_2/ReadVariableOp?!sequential_9/conv_lst_m2d_9/while?,sequential_9/dense_18/BiasAdd/ReadVariableOp?+sequential_9/dense_18/MatMul/ReadVariableOp?,sequential_9/dense_19/BiasAdd/ReadVariableOp?+sequential_9/dense_19/MatMul/ReadVariableOp?
&sequential_9/conv_lst_m2d_9/zeros_like	ZerosLikeconv_lst_m2d_9_input*
T0*3
_output_shapes!
:?????????d	2(
&sequential_9/conv_lst_m2d_9/zeros_like?
1sequential_9/conv_lst_m2d_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/conv_lst_m2d_9/Sum/reduction_indices?
sequential_9/conv_lst_m2d_9/SumSum*sequential_9/conv_lst_m2d_9/zeros_like:y:0:sequential_9/conv_lst_m2d_9/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2!
sequential_9/conv_lst_m2d_9/Sum?
1sequential_9/conv_lst_m2d_9/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   23
1sequential_9/conv_lst_m2d_9/zeros/shape_as_tensor?
'sequential_9/conv_lst_m2d_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'sequential_9/conv_lst_m2d_9/zeros/Const?
!sequential_9/conv_lst_m2d_9/zerosFill:sequential_9/conv_lst_m2d_9/zeros/shape_as_tensor:output:00sequential_9/conv_lst_m2d_9/zeros/Const:output:0*
T0*&
_output_shapes
:	@2#
!sequential_9/conv_lst_m2d_9/zeros?
'sequential_9/conv_lst_m2d_9/convolutionConv2D(sequential_9/conv_lst_m2d_9/Sum:output:0*sequential_9/conv_lst_m2d_9/zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2)
'sequential_9/conv_lst_m2d_9/convolution?
*sequential_9/conv_lst_m2d_9/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2,
*sequential_9/conv_lst_m2d_9/transpose/perm?
%sequential_9/conv_lst_m2d_9/transpose	Transposeconv_lst_m2d_9_input3sequential_9/conv_lst_m2d_9/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????d	2'
%sequential_9/conv_lst_m2d_9/transpose?
!sequential_9/conv_lst_m2d_9/ShapeShape)sequential_9/conv_lst_m2d_9/transpose:y:0*
T0*
_output_shapes
:2#
!sequential_9/conv_lst_m2d_9/Shape?
/sequential_9/conv_lst_m2d_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_9/conv_lst_m2d_9/strided_slice/stack?
1sequential_9/conv_lst_m2d_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_9/conv_lst_m2d_9/strided_slice/stack_1?
1sequential_9/conv_lst_m2d_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential_9/conv_lst_m2d_9/strided_slice/stack_2?
)sequential_9/conv_lst_m2d_9/strided_sliceStridedSlice*sequential_9/conv_lst_m2d_9/Shape:output:08sequential_9/conv_lst_m2d_9/strided_slice/stack:output:0:sequential_9/conv_lst_m2d_9/strided_slice/stack_1:output:0:sequential_9/conv_lst_m2d_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential_9/conv_lst_m2d_9/strided_slice?
7sequential_9/conv_lst_m2d_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7sequential_9/conv_lst_m2d_9/TensorArrayV2/element_shape?
)sequential_9/conv_lst_m2d_9/TensorArrayV2TensorListReserve@sequential_9/conv_lst_m2d_9/TensorArrayV2/element_shape:output:02sequential_9/conv_lst_m2d_9/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02+
)sequential_9/conv_lst_m2d_9/TensorArrayV2?
Qsequential_9/conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   2S
Qsequential_9/conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
Csequential_9/conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor)sequential_9/conv_lst_m2d_9/transpose:y:0Zsequential_9/conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02E
Csequential_9/conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor?
1sequential_9/conv_lst_m2d_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_9/conv_lst_m2d_9/strided_slice_1/stack?
3sequential_9/conv_lst_m2d_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_9/conv_lst_m2d_9/strided_slice_1/stack_1?
3sequential_9/conv_lst_m2d_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_9/conv_lst_m2d_9/strided_slice_1/stack_2?
+sequential_9/conv_lst_m2d_9/strided_slice_1StridedSlice)sequential_9/conv_lst_m2d_9/transpose:y:0:sequential_9/conv_lst_m2d_9/strided_slice_1/stack:output:0<sequential_9/conv_lst_m2d_9/strided_slice_1/stack_1:output:0<sequential_9/conv_lst_m2d_9/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2-
+sequential_9/conv_lst_m2d_9/strided_slice_1?
+sequential_9/conv_lst_m2d_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_9/conv_lst_m2d_9/split/split_dim?
0sequential_9/conv_lst_m2d_9/split/ReadVariableOpReadVariableOp9sequential_9_conv_lst_m2d_9_split_readvariableop_resource*'
_output_shapes
:	?*
dtype022
0sequential_9/conv_lst_m2d_9/split/ReadVariableOp?
!sequential_9/conv_lst_m2d_9/splitSplit4sequential_9/conv_lst_m2d_9/split/split_dim:output:08sequential_9/conv_lst_m2d_9/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2#
!sequential_9/conv_lst_m2d_9/split?
-sequential_9/conv_lst_m2d_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-sequential_9/conv_lst_m2d_9/split_1/split_dim?
2sequential_9/conv_lst_m2d_9/split_1/ReadVariableOpReadVariableOp;sequential_9_conv_lst_m2d_9_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype024
2sequential_9/conv_lst_m2d_9/split_1/ReadVariableOp?
#sequential_9/conv_lst_m2d_9/split_1Split6sequential_9/conv_lst_m2d_9/split_1/split_dim:output:0:sequential_9/conv_lst_m2d_9/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2%
#sequential_9/conv_lst_m2d_9/split_1?
-sequential_9/conv_lst_m2d_9/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_9/conv_lst_m2d_9/split_2/split_dim?
2sequential_9/conv_lst_m2d_9/split_2/ReadVariableOpReadVariableOp;sequential_9_conv_lst_m2d_9_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype024
2sequential_9/conv_lst_m2d_9/split_2/ReadVariableOp?
#sequential_9/conv_lst_m2d_9/split_2Split6sequential_9/conv_lst_m2d_9/split_2/split_dim:output:0:sequential_9/conv_lst_m2d_9/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2%
#sequential_9/conv_lst_m2d_9/split_2?
)sequential_9/conv_lst_m2d_9/convolution_1Conv2D4sequential_9/conv_lst_m2d_9/strided_slice_1:output:0*sequential_9/conv_lst_m2d_9/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_1?
#sequential_9/conv_lst_m2d_9/BiasAddBiasAdd2sequential_9/conv_lst_m2d_9/convolution_1:output:0,sequential_9/conv_lst_m2d_9/split_2:output:0*
T0*/
_output_shapes
:?????????b@2%
#sequential_9/conv_lst_m2d_9/BiasAdd?
)sequential_9/conv_lst_m2d_9/convolution_2Conv2D4sequential_9/conv_lst_m2d_9/strided_slice_1:output:0*sequential_9/conv_lst_m2d_9/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_2?
%sequential_9/conv_lst_m2d_9/BiasAdd_1BiasAdd2sequential_9/conv_lst_m2d_9/convolution_2:output:0,sequential_9/conv_lst_m2d_9/split_2:output:1*
T0*/
_output_shapes
:?????????b@2'
%sequential_9/conv_lst_m2d_9/BiasAdd_1?
)sequential_9/conv_lst_m2d_9/convolution_3Conv2D4sequential_9/conv_lst_m2d_9/strided_slice_1:output:0*sequential_9/conv_lst_m2d_9/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_3?
%sequential_9/conv_lst_m2d_9/BiasAdd_2BiasAdd2sequential_9/conv_lst_m2d_9/convolution_3:output:0,sequential_9/conv_lst_m2d_9/split_2:output:2*
T0*/
_output_shapes
:?????????b@2'
%sequential_9/conv_lst_m2d_9/BiasAdd_2?
)sequential_9/conv_lst_m2d_9/convolution_4Conv2D4sequential_9/conv_lst_m2d_9/strided_slice_1:output:0*sequential_9/conv_lst_m2d_9/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_4?
%sequential_9/conv_lst_m2d_9/BiasAdd_3BiasAdd2sequential_9/conv_lst_m2d_9/convolution_4:output:0,sequential_9/conv_lst_m2d_9/split_2:output:3*
T0*/
_output_shapes
:?????????b@2'
%sequential_9/conv_lst_m2d_9/BiasAdd_3?
)sequential_9/conv_lst_m2d_9/convolution_5Conv2D0sequential_9/conv_lst_m2d_9/convolution:output:0,sequential_9/conv_lst_m2d_9/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_5?
)sequential_9/conv_lst_m2d_9/convolution_6Conv2D0sequential_9/conv_lst_m2d_9/convolution:output:0,sequential_9/conv_lst_m2d_9/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_6?
)sequential_9/conv_lst_m2d_9/convolution_7Conv2D0sequential_9/conv_lst_m2d_9/convolution:output:0,sequential_9/conv_lst_m2d_9/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_7?
)sequential_9/conv_lst_m2d_9/convolution_8Conv2D0sequential_9/conv_lst_m2d_9/convolution:output:0,sequential_9/conv_lst_m2d_9/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2+
)sequential_9/conv_lst_m2d_9/convolution_8?
sequential_9/conv_lst_m2d_9/addAddV2,sequential_9/conv_lst_m2d_9/BiasAdd:output:02sequential_9/conv_lst_m2d_9/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2!
sequential_9/conv_lst_m2d_9/add?
!sequential_9/conv_lst_m2d_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!sequential_9/conv_lst_m2d_9/Const?
#sequential_9/conv_lst_m2d_9/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential_9/conv_lst_m2d_9/Const_1?
sequential_9/conv_lst_m2d_9/MulMul#sequential_9/conv_lst_m2d_9/add:z:0*sequential_9/conv_lst_m2d_9/Const:output:0*
T0*/
_output_shapes
:?????????b@2!
sequential_9/conv_lst_m2d_9/Mul?
!sequential_9/conv_lst_m2d_9/Add_1Add#sequential_9/conv_lst_m2d_9/Mul:z:0,sequential_9/conv_lst_m2d_9/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/Add_1?
3sequential_9/conv_lst_m2d_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??25
3sequential_9/conv_lst_m2d_9/clip_by_value/Minimum/y?
1sequential_9/conv_lst_m2d_9/clip_by_value/MinimumMinimum%sequential_9/conv_lst_m2d_9/Add_1:z:0<sequential_9/conv_lst_m2d_9/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@23
1sequential_9/conv_lst_m2d_9/clip_by_value/Minimum?
+sequential_9/conv_lst_m2d_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential_9/conv_lst_m2d_9/clip_by_value/y?
)sequential_9/conv_lst_m2d_9/clip_by_valueMaximum5sequential_9/conv_lst_m2d_9/clip_by_value/Minimum:z:04sequential_9/conv_lst_m2d_9/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2+
)sequential_9/conv_lst_m2d_9/clip_by_value?
!sequential_9/conv_lst_m2d_9/add_2AddV2.sequential_9/conv_lst_m2d_9/BiasAdd_1:output:02sequential_9/conv_lst_m2d_9/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/add_2?
#sequential_9/conv_lst_m2d_9/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2%
#sequential_9/conv_lst_m2d_9/Const_2?
#sequential_9/conv_lst_m2d_9/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential_9/conv_lst_m2d_9/Const_3?
!sequential_9/conv_lst_m2d_9/Mul_1Mul%sequential_9/conv_lst_m2d_9/add_2:z:0,sequential_9/conv_lst_m2d_9/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/Mul_1?
!sequential_9/conv_lst_m2d_9/Add_3Add%sequential_9/conv_lst_m2d_9/Mul_1:z:0,sequential_9/conv_lst_m2d_9/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/Add_3?
5sequential_9/conv_lst_m2d_9/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5sequential_9/conv_lst_m2d_9/clip_by_value_1/Minimum/y?
3sequential_9/conv_lst_m2d_9/clip_by_value_1/MinimumMinimum%sequential_9/conv_lst_m2d_9/Add_3:z:0>sequential_9/conv_lst_m2d_9/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@25
3sequential_9/conv_lst_m2d_9/clip_by_value_1/Minimum?
-sequential_9/conv_lst_m2d_9/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_9/conv_lst_m2d_9/clip_by_value_1/y?
+sequential_9/conv_lst_m2d_9/clip_by_value_1Maximum7sequential_9/conv_lst_m2d_9/clip_by_value_1/Minimum:z:06sequential_9/conv_lst_m2d_9/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2-
+sequential_9/conv_lst_m2d_9/clip_by_value_1?
!sequential_9/conv_lst_m2d_9/mul_2Mul/sequential_9/conv_lst_m2d_9/clip_by_value_1:z:00sequential_9/conv_lst_m2d_9/convolution:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/mul_2?
!sequential_9/conv_lst_m2d_9/add_4AddV2.sequential_9/conv_lst_m2d_9/BiasAdd_2:output:02sequential_9/conv_lst_m2d_9/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/add_4?
 sequential_9/conv_lst_m2d_9/ReluRelu%sequential_9/conv_lst_m2d_9/add_4:z:0*
T0*/
_output_shapes
:?????????b@2"
 sequential_9/conv_lst_m2d_9/Relu?
!sequential_9/conv_lst_m2d_9/mul_3Mul-sequential_9/conv_lst_m2d_9/clip_by_value:z:0.sequential_9/conv_lst_m2d_9/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/mul_3?
!sequential_9/conv_lst_m2d_9/add_5AddV2%sequential_9/conv_lst_m2d_9/mul_2:z:0%sequential_9/conv_lst_m2d_9/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/add_5?
!sequential_9/conv_lst_m2d_9/add_6AddV2.sequential_9/conv_lst_m2d_9/BiasAdd_3:output:02sequential_9/conv_lst_m2d_9/convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/add_6?
#sequential_9/conv_lst_m2d_9/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2%
#sequential_9/conv_lst_m2d_9/Const_4?
#sequential_9/conv_lst_m2d_9/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2%
#sequential_9/conv_lst_m2d_9/Const_5?
!sequential_9/conv_lst_m2d_9/Mul_4Mul%sequential_9/conv_lst_m2d_9/add_6:z:0,sequential_9/conv_lst_m2d_9/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/Mul_4?
!sequential_9/conv_lst_m2d_9/Add_7Add%sequential_9/conv_lst_m2d_9/Mul_4:z:0,sequential_9/conv_lst_m2d_9/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/Add_7?
5sequential_9/conv_lst_m2d_9/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5sequential_9/conv_lst_m2d_9/clip_by_value_2/Minimum/y?
3sequential_9/conv_lst_m2d_9/clip_by_value_2/MinimumMinimum%sequential_9/conv_lst_m2d_9/Add_7:z:0>sequential_9/conv_lst_m2d_9/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@25
3sequential_9/conv_lst_m2d_9/clip_by_value_2/Minimum?
-sequential_9/conv_lst_m2d_9/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_9/conv_lst_m2d_9/clip_by_value_2/y?
+sequential_9/conv_lst_m2d_9/clip_by_value_2Maximum7sequential_9/conv_lst_m2d_9/clip_by_value_2/Minimum:z:06sequential_9/conv_lst_m2d_9/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2-
+sequential_9/conv_lst_m2d_9/clip_by_value_2?
"sequential_9/conv_lst_m2d_9/Relu_1Relu%sequential_9/conv_lst_m2d_9/add_5:z:0*
T0*/
_output_shapes
:?????????b@2$
"sequential_9/conv_lst_m2d_9/Relu_1?
!sequential_9/conv_lst_m2d_9/mul_5Mul/sequential_9/conv_lst_m2d_9/clip_by_value_2:z:00sequential_9/conv_lst_m2d_9/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2#
!sequential_9/conv_lst_m2d_9/mul_5?
9sequential_9/conv_lst_m2d_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2;
9sequential_9/conv_lst_m2d_9/TensorArrayV2_1/element_shape?
+sequential_9/conv_lst_m2d_9/TensorArrayV2_1TensorListReserveBsequential_9/conv_lst_m2d_9/TensorArrayV2_1/element_shape:output:02sequential_9/conv_lst_m2d_9/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+sequential_9/conv_lst_m2d_9/TensorArrayV2_1?
 sequential_9/conv_lst_m2d_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_9/conv_lst_m2d_9/time?
4sequential_9/conv_lst_m2d_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4sequential_9/conv_lst_m2d_9/while/maximum_iterations?
.sequential_9/conv_lst_m2d_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_9/conv_lst_m2d_9/while/loop_counter?	
!sequential_9/conv_lst_m2d_9/whileWhile7sequential_9/conv_lst_m2d_9/while/loop_counter:output:0=sequential_9/conv_lst_m2d_9/while/maximum_iterations:output:0)sequential_9/conv_lst_m2d_9/time:output:04sequential_9/conv_lst_m2d_9/TensorArrayV2_1:handle:00sequential_9/conv_lst_m2d_9/convolution:output:00sequential_9/conv_lst_m2d_9/convolution:output:02sequential_9/conv_lst_m2d_9/strided_slice:output:0Ssequential_9/conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor:output_handle:09sequential_9_conv_lst_m2d_9_split_readvariableop_resource;sequential_9_conv_lst_m2d_9_split_1_readvariableop_resource;sequential_9_conv_lst_m2d_9_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*8
body0R.
,sequential_9_conv_lst_m2d_9_while_body_60461*8
cond0R.
,sequential_9_conv_lst_m2d_9_while_cond_60460*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2#
!sequential_9/conv_lst_m2d_9/while?
Lsequential_9/conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2N
Lsequential_9/conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shape?
>sequential_9/conv_lst_m2d_9/TensorArrayV2Stack/TensorListStackTensorListStack*sequential_9/conv_lst_m2d_9/while:output:3Usequential_9/conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????b@*
element_dtype02@
>sequential_9/conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack?
1sequential_9/conv_lst_m2d_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1sequential_9/conv_lst_m2d_9/strided_slice_2/stack?
3sequential_9/conv_lst_m2d_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_9/conv_lst_m2d_9/strided_slice_2/stack_1?
3sequential_9/conv_lst_m2d_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_9/conv_lst_m2d_9/strided_slice_2/stack_2?
+sequential_9/conv_lst_m2d_9/strided_slice_2StridedSliceGsequential_9/conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack:tensor:0:sequential_9/conv_lst_m2d_9/strided_slice_2/stack:output:0<sequential_9/conv_lst_m2d_9/strided_slice_2/stack_1:output:0<sequential_9/conv_lst_m2d_9/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2-
+sequential_9/conv_lst_m2d_9/strided_slice_2?
,sequential_9/conv_lst_m2d_9/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2.
,sequential_9/conv_lst_m2d_9/transpose_1/perm?
'sequential_9/conv_lst_m2d_9/transpose_1	TransposeGsequential_9/conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack:tensor:05sequential_9/conv_lst_m2d_9/transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????b@2)
'sequential_9/conv_lst_m2d_9/transpose_1?
sequential_9/dropout_9/IdentityIdentity4sequential_9/conv_lst_m2d_9/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????b@2!
sequential_9/dropout_9/Identity?
sequential_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
sequential_9/flatten_9/Const?
sequential_9/flatten_9/ReshapeReshape(sequential_9/dropout_9/Identity:output:0%sequential_9/flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????12 
sequential_9/flatten_9/Reshape?
+sequential_9/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_18_matmul_readvariableop_resource*
_output_shapes
:	?1d*
dtype02-
+sequential_9/dense_18/MatMul/ReadVariableOp?
sequential_9/dense_18/MatMulMatMul'sequential_9/flatten_9/Reshape:output:03sequential_9/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_9/dense_18/MatMul?
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,sequential_9/dense_18/BiasAdd/ReadVariableOp?
sequential_9/dense_18/BiasAddBiasAdd&sequential_9/dense_18/MatMul:product:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_9/dense_18/BiasAdd?
sequential_9/dense_18/ReluRelu&sequential_9/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_9/dense_18/Relu?
+sequential_9/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_19_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02-
+sequential_9/dense_19/MatMul/ReadVariableOp?
sequential_9/dense_19/MatMulMatMul(sequential_9/dense_18/Relu:activations:03sequential_9/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_19/MatMul?
,sequential_9/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_9/dense_19/BiasAdd/ReadVariableOp?
sequential_9/dense_19/BiasAddBiasAdd&sequential_9/dense_19/MatMul:product:04sequential_9/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_19/BiasAdd?
sequential_9/dense_19/SoftmaxSoftmax&sequential_9/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_9/dense_19/Softmax?
IdentityIdentity'sequential_9/dense_19/Softmax:softmax:01^sequential_9/conv_lst_m2d_9/split/ReadVariableOp3^sequential_9/conv_lst_m2d_9/split_1/ReadVariableOp3^sequential_9/conv_lst_m2d_9/split_2/ReadVariableOp"^sequential_9/conv_lst_m2d_9/while-^sequential_9/dense_18/BiasAdd/ReadVariableOp,^sequential_9/dense_18/MatMul/ReadVariableOp-^sequential_9/dense_19/BiasAdd/ReadVariableOp,^sequential_9/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 2d
0sequential_9/conv_lst_m2d_9/split/ReadVariableOp0sequential_9/conv_lst_m2d_9/split/ReadVariableOp2h
2sequential_9/conv_lst_m2d_9/split_1/ReadVariableOp2sequential_9/conv_lst_m2d_9/split_1/ReadVariableOp2h
2sequential_9/conv_lst_m2d_9/split_2/ReadVariableOp2sequential_9/conv_lst_m2d_9/split_2/ReadVariableOp2F
!sequential_9/conv_lst_m2d_9/while!sequential_9/conv_lst_m2d_9/while2\
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_18/MatMul/ReadVariableOp+sequential_9/dense_18/MatMul/ReadVariableOp2\
,sequential_9/dense_19/BiasAdd/ReadVariableOp,sequential_9/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_19/MatMul/ReadVariableOp+sequential_9/dense_19/MatMul/ReadVariableOp:i e
3
_output_shapes!
:?????????d	
.
_user_specified_nameconv_lst_m2d_9_input
?h
?
while_body_61721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:	?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:	?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1|
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_4k

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:?????????b@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2o
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9?
while/IdentityIdentitywhile/add_9:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add_8:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
??
?
conv_lst_m2d_9_while_body_62143:
6conv_lst_m2d_9_while_conv_lst_m2d_9_while_loop_counter@
<conv_lst_m2d_9_while_conv_lst_m2d_9_while_maximum_iterations$
 conv_lst_m2d_9_while_placeholder&
"conv_lst_m2d_9_while_placeholder_1&
"conv_lst_m2d_9_while_placeholder_2&
"conv_lst_m2d_9_while_placeholder_37
3conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slice_0u
qconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0O
4conv_lst_m2d_9_while_split_readvariableop_resource_0:	?Q
6conv_lst_m2d_9_while_split_1_readvariableop_resource_0:@?E
6conv_lst_m2d_9_while_split_2_readvariableop_resource_0:	?!
conv_lst_m2d_9_while_identity#
conv_lst_m2d_9_while_identity_1#
conv_lst_m2d_9_while_identity_2#
conv_lst_m2d_9_while_identity_3#
conv_lst_m2d_9_while_identity_4#
conv_lst_m2d_9_while_identity_55
1conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slices
oconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensorM
2conv_lst_m2d_9_while_split_readvariableop_resource:	?O
4conv_lst_m2d_9_while_split_1_readvariableop_resource:@?C
4conv_lst_m2d_9_while_split_2_readvariableop_resource:	???)conv_lst_m2d_9/while/split/ReadVariableOp?+conv_lst_m2d_9/while/split_1/ReadVariableOp?+conv_lst_m2d_9/while/split_2/ReadVariableOp?
Fconv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   2H
Fconv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0 conv_lst_m2d_9_while_placeholderOconv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02:
8conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem?
$conv_lst_m2d_9/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_9/while/split/split_dim?
)conv_lst_m2d_9/while/split/ReadVariableOpReadVariableOp4conv_lst_m2d_9_while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02+
)conv_lst_m2d_9/while/split/ReadVariableOp?
conv_lst_m2d_9/while/splitSplit-conv_lst_m2d_9/while/split/split_dim:output:01conv_lst_m2d_9/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
conv_lst_m2d_9/while/split?
&conv_lst_m2d_9/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&conv_lst_m2d_9/while/split_1/split_dim?
+conv_lst_m2d_9/while/split_1/ReadVariableOpReadVariableOp6conv_lst_m2d_9_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02-
+conv_lst_m2d_9/while/split_1/ReadVariableOp?
conv_lst_m2d_9/while/split_1Split/conv_lst_m2d_9/while/split_1/split_dim:output:03conv_lst_m2d_9/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_9/while/split_1?
&conv_lst_m2d_9/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&conv_lst_m2d_9/while/split_2/split_dim?
+conv_lst_m2d_9/while/split_2/ReadVariableOpReadVariableOp6conv_lst_m2d_9_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02-
+conv_lst_m2d_9/while/split_2/ReadVariableOp?
conv_lst_m2d_9/while/split_2Split/conv_lst_m2d_9/while/split_2/split_dim:output:03conv_lst_m2d_9/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_9/while/split_2?
 conv_lst_m2d_9/while/convolutionConv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2"
 conv_lst_m2d_9/while/convolution?
conv_lst_m2d_9/while/BiasAddBiasAdd)conv_lst_m2d_9/while/convolution:output:0%conv_lst_m2d_9/while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/BiasAdd?
"conv_lst_m2d_9/while/convolution_1Conv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2$
"conv_lst_m2d_9/while/convolution_1?
conv_lst_m2d_9/while/BiasAdd_1BiasAdd+conv_lst_m2d_9/while/convolution_1:output:0%conv_lst_m2d_9/while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/while/BiasAdd_1?
"conv_lst_m2d_9/while/convolution_2Conv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2$
"conv_lst_m2d_9/while/convolution_2?
conv_lst_m2d_9/while/BiasAdd_2BiasAdd+conv_lst_m2d_9/while/convolution_2:output:0%conv_lst_m2d_9/while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/while/BiasAdd_2?
"conv_lst_m2d_9/while/convolution_3Conv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2$
"conv_lst_m2d_9/while/convolution_3?
conv_lst_m2d_9/while/BiasAdd_3BiasAdd+conv_lst_m2d_9/while/convolution_3:output:0%conv_lst_m2d_9/while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/while/BiasAdd_3?
"conv_lst_m2d_9/while/convolution_4Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_4?
"conv_lst_m2d_9/while/convolution_5Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_5?
"conv_lst_m2d_9/while/convolution_6Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_6?
"conv_lst_m2d_9/while/convolution_7Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_7?
conv_lst_m2d_9/while/addAddV2%conv_lst_m2d_9/while/BiasAdd:output:0+conv_lst_m2d_9/while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add}
conv_lst_m2d_9/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/while/Const?
conv_lst_m2d_9/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/while/Const_1?
conv_lst_m2d_9/while/MulMulconv_lst_m2d_9/while/add:z:0#conv_lst_m2d_9/while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Mul?
conv_lst_m2d_9/while/Add_1Addconv_lst_m2d_9/while/Mul:z:0%conv_lst_m2d_9/while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Add_1?
,conv_lst_m2d_9/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d_9/while/clip_by_value/Minimum/y?
*conv_lst_m2d_9/while/clip_by_value/MinimumMinimumconv_lst_m2d_9/while/Add_1:z:05conv_lst_m2d_9/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2,
*conv_lst_m2d_9/while/clip_by_value/Minimum?
$conv_lst_m2d_9/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d_9/while/clip_by_value/y?
"conv_lst_m2d_9/while/clip_by_valueMaximum.conv_lst_m2d_9/while/clip_by_value/Minimum:z:0-conv_lst_m2d_9/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2$
"conv_lst_m2d_9/while/clip_by_value?
conv_lst_m2d_9/while/add_2AddV2'conv_lst_m2d_9/while/BiasAdd_1:output:0+conv_lst_m2d_9/while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_2?
conv_lst_m2d_9/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/while/Const_2?
conv_lst_m2d_9/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/while/Const_3?
conv_lst_m2d_9/while/Mul_1Mulconv_lst_m2d_9/while/add_2:z:0%conv_lst_m2d_9/while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Mul_1?
conv_lst_m2d_9/while/Add_3Addconv_lst_m2d_9/while/Mul_1:z:0%conv_lst_m2d_9/while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Add_3?
.conv_lst_m2d_9/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_9/while/clip_by_value_1/Minimum/y?
,conv_lst_m2d_9/while/clip_by_value_1/MinimumMinimumconv_lst_m2d_9/while/Add_3:z:07conv_lst_m2d_9/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2.
,conv_lst_m2d_9/while/clip_by_value_1/Minimum?
&conv_lst_m2d_9/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_9/while/clip_by_value_1/y?
$conv_lst_m2d_9/while/clip_by_value_1Maximum0conv_lst_m2d_9/while/clip_by_value_1/Minimum:z:0/conv_lst_m2d_9/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2&
$conv_lst_m2d_9/while/clip_by_value_1?
conv_lst_m2d_9/while/mul_2Mul(conv_lst_m2d_9/while/clip_by_value_1:z:0"conv_lst_m2d_9_while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/mul_2?
conv_lst_m2d_9/while/add_4AddV2'conv_lst_m2d_9/while/BiasAdd_2:output:0+conv_lst_m2d_9/while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_4?
conv_lst_m2d_9/while/ReluReluconv_lst_m2d_9/while/add_4:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Relu?
conv_lst_m2d_9/while/mul_3Mul&conv_lst_m2d_9/while/clip_by_value:z:0'conv_lst_m2d_9/while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/mul_3?
conv_lst_m2d_9/while/add_5AddV2conv_lst_m2d_9/while/mul_2:z:0conv_lst_m2d_9/while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_5?
conv_lst_m2d_9/while/add_6AddV2'conv_lst_m2d_9/while/BiasAdd_3:output:0+conv_lst_m2d_9/while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_6?
conv_lst_m2d_9/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/while/Const_4?
conv_lst_m2d_9/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/while/Const_5?
conv_lst_m2d_9/while/Mul_4Mulconv_lst_m2d_9/while/add_6:z:0%conv_lst_m2d_9/while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Mul_4?
conv_lst_m2d_9/while/Add_7Addconv_lst_m2d_9/while/Mul_4:z:0%conv_lst_m2d_9/while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Add_7?
.conv_lst_m2d_9/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_9/while/clip_by_value_2/Minimum/y?
,conv_lst_m2d_9/while/clip_by_value_2/MinimumMinimumconv_lst_m2d_9/while/Add_7:z:07conv_lst_m2d_9/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2.
,conv_lst_m2d_9/while/clip_by_value_2/Minimum?
&conv_lst_m2d_9/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_9/while/clip_by_value_2/y?
$conv_lst_m2d_9/while/clip_by_value_2Maximum0conv_lst_m2d_9/while/clip_by_value_2/Minimum:z:0/conv_lst_m2d_9/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2&
$conv_lst_m2d_9/while/clip_by_value_2?
conv_lst_m2d_9/while/Relu_1Reluconv_lst_m2d_9/while/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Relu_1?
conv_lst_m2d_9/while/mul_5Mul(conv_lst_m2d_9/while/clip_by_value_2:z:0)conv_lst_m2d_9/while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/mul_5?
9conv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"conv_lst_m2d_9_while_placeholder_1 conv_lst_m2d_9_while_placeholderconv_lst_m2d_9/while/mul_5:z:0*
_output_shapes
: *
element_dtype02;
9conv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItem~
conv_lst_m2d_9/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_9/while/add_8/y?
conv_lst_m2d_9/while/add_8AddV2 conv_lst_m2d_9_while_placeholder%conv_lst_m2d_9/while/add_8/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/add_8~
conv_lst_m2d_9/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_9/while/add_9/y?
conv_lst_m2d_9/while/add_9AddV26conv_lst_m2d_9_while_conv_lst_m2d_9_while_loop_counter%conv_lst_m2d_9/while/add_9/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/add_9?
conv_lst_m2d_9/while/IdentityIdentityconv_lst_m2d_9/while/add_9:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/Identity?
conv_lst_m2d_9/while/Identity_1Identity<conv_lst_m2d_9_while_conv_lst_m2d_9_while_maximum_iterations*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_9/while/Identity_1?
conv_lst_m2d_9/while/Identity_2Identityconv_lst_m2d_9/while/add_8:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_9/while/Identity_2?
conv_lst_m2d_9/while/Identity_3IdentityIconv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_9/while/Identity_3?
conv_lst_m2d_9/while/Identity_4Identityconv_lst_m2d_9/while/mul_5:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2!
conv_lst_m2d_9/while/Identity_4?
conv_lst_m2d_9/while/Identity_5Identityconv_lst_m2d_9/while/add_5:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2!
conv_lst_m2d_9/while/Identity_5"h
1conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slice3conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slice_0"G
conv_lst_m2d_9_while_identity&conv_lst_m2d_9/while/Identity:output:0"K
conv_lst_m2d_9_while_identity_1(conv_lst_m2d_9/while/Identity_1:output:0"K
conv_lst_m2d_9_while_identity_2(conv_lst_m2d_9/while/Identity_2:output:0"K
conv_lst_m2d_9_while_identity_3(conv_lst_m2d_9/while/Identity_3:output:0"K
conv_lst_m2d_9_while_identity_4(conv_lst_m2d_9/while/Identity_4:output:0"K
conv_lst_m2d_9_while_identity_5(conv_lst_m2d_9/while/Identity_5:output:0"n
4conv_lst_m2d_9_while_split_1_readvariableop_resource6conv_lst_m2d_9_while_split_1_readvariableop_resource_0"n
4conv_lst_m2d_9_while_split_2_readvariableop_resource6conv_lst_m2d_9_while_split_2_readvariableop_resource_0"j
2conv_lst_m2d_9_while_split_readvariableop_resource4conv_lst_m2d_9_while_split_readvariableop_resource_0"?
oconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensorqconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 2V
)conv_lst_m2d_9/while/split/ReadVariableOp)conv_lst_m2d_9/while/split/ReadVariableOp2Z
+conv_lst_m2d_9/while/split_1/ReadVariableOp+conv_lst_m2d_9/while/split_1/ReadVariableOp2Z
+conv_lst_m2d_9/while/split_2/ReadVariableOp+conv_lst_m2d_9/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_63337
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_63337___redundant_placeholder03
/while_while_cond_63337___redundant_placeholder13
/while_while_cond_63337___redundant_placeholder23
/while_while_cond_63337___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?#
?
while_body_60722
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_60746_0:	?(
while_60748_0:@?
while_60750_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_60746:	?&
while_60748:@?
while_60750:	???while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_60746_0while_60748_0while_60750_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????b@:?????????b@:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_607082
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1~
while/IdentityIdentitywhile/add_1:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2
while/Identity_5"
while_60746while_60746_0"
while_60748while_60748_0"
while_60750while_60750_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_62893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_62893___redundant_placeholder03
/while_while_cond_62893___redundant_placeholder13
/while_while_cond_62893___redundant_placeholder23
/while_while_cond_62893___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
??
?
!__inference__traced_restore_63939
file_prefix3
 assignvariableop_dense_18_kernel:	?1d.
 assignvariableop_1_dense_18_bias:d4
"assignvariableop_2_dense_19_kernel:d.
 assignvariableop_3_dense_19_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: C
(assignvariableop_9_conv_lst_m2d_9_kernel:	?N
3assignvariableop_10_conv_lst_m2d_9_recurrent_kernel:@?6
'assignvariableop_11_conv_lst_m2d_9_bias:	?#
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: =
*assignvariableop_16_adam_dense_18_kernel_m:	?1d6
(assignvariableop_17_adam_dense_18_bias_m:d<
*assignvariableop_18_adam_dense_19_kernel_m:d6
(assignvariableop_19_adam_dense_19_bias_m:K
0assignvariableop_20_adam_conv_lst_m2d_9_kernel_m:	?U
:assignvariableop_21_adam_conv_lst_m2d_9_recurrent_kernel_m:@?=
.assignvariableop_22_adam_conv_lst_m2d_9_bias_m:	?=
*assignvariableop_23_adam_dense_18_kernel_v:	?1d6
(assignvariableop_24_adam_dense_18_bias_v:d<
*assignvariableop_25_adam_dense_19_kernel_v:d6
(assignvariableop_26_adam_dense_19_bias_v:K
0assignvariableop_27_adam_conv_lst_m2d_9_kernel_v:	?U
:assignvariableop_28_adam_conv_lst_m2d_9_recurrent_kernel_v:@?=
.assignvariableop_29_adam_conv_lst_m2d_9_bias_v:	?
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_19_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_conv_lst_m2d_9_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_conv_lst_m2d_9_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_conv_lst_m2d_9_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_18_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_18_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_19_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_19_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_conv_lst_m2d_9_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp:assignvariableop_21_adam_conv_lst_m2d_9_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_conv_lst_m2d_9_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_18_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_18_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_19_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_19_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_conv_lst_m2d_9_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp:assignvariableop_28_adam_conv_lst_m2d_9_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_conv_lst_m2d_9_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_299
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30?
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?h
?
while_body_63116
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:	?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:	?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1|
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_4k

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:?????????b@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2o
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9?
while/IdentityIdentitywhile/add_9:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add_8:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?h
?
while_body_63338
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:	?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:	?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1|
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_4k

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:?????????b@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2o
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9?
while/IdentityIdentitywhile/add_9:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add_8:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?
?
conv_lst_m2d_9_while_cond_62142:
6conv_lst_m2d_9_while_conv_lst_m2d_9_while_loop_counter@
<conv_lst_m2d_9_while_conv_lst_m2d_9_while_maximum_iterations$
 conv_lst_m2d_9_while_placeholder&
"conv_lst_m2d_9_while_placeholder_1&
"conv_lst_m2d_9_while_placeholder_2&
"conv_lst_m2d_9_while_placeholder_3:
6conv_lst_m2d_9_while_less_conv_lst_m2d_9_strided_sliceQ
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62142___redundant_placeholder0Q
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62142___redundant_placeholder1Q
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62142___redundant_placeholder2Q
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62142___redundant_placeholder3!
conv_lst_m2d_9_while_identity
?
conv_lst_m2d_9/while/LessLess conv_lst_m2d_9_while_placeholder6conv_lst_m2d_9_while_less_conv_lst_m2d_9_strided_slice*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/Less?
conv_lst_m2d_9/while/IdentityIdentityconv_lst_m2d_9/while/Less:z:0*
T0
*
_output_shapes
: 2
conv_lst_m2d_9/while/Identity"G
conv_lst_m2d_9_while_identity&conv_lst_m2d_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_62671
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_62671___redundant_placeholder03
/while_while_cond_62671___redundant_placeholder13
/while_while_cond_62671___redundant_placeholder23
/while_while_cond_62671___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?8
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_60790

inputs"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?whilet

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&??????????????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????b@:?????????b@:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_607082
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0	unknown_1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_60722*
condR
while_cond_60721*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^StatefulPartitionedCall^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&??????????????????d	: : : 22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&??????????????????d	
 
_user_specified_nameinputs
?

?
C__inference_dense_19_layer_call_and_return_conditional_losses_61539

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
while_cond_60959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_60959___redundant_placeholder03
/while_while_cond_60959___redundant_placeholder13
/while_while_cond_60959___redundant_placeholder23
/while_while_cond_60959___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
??
?
conv_lst_m2d_9_while_body_62382:
6conv_lst_m2d_9_while_conv_lst_m2d_9_while_loop_counter@
<conv_lst_m2d_9_while_conv_lst_m2d_9_while_maximum_iterations$
 conv_lst_m2d_9_while_placeholder&
"conv_lst_m2d_9_while_placeholder_1&
"conv_lst_m2d_9_while_placeholder_2&
"conv_lst_m2d_9_while_placeholder_37
3conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slice_0u
qconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0O
4conv_lst_m2d_9_while_split_readvariableop_resource_0:	?Q
6conv_lst_m2d_9_while_split_1_readvariableop_resource_0:@?E
6conv_lst_m2d_9_while_split_2_readvariableop_resource_0:	?!
conv_lst_m2d_9_while_identity#
conv_lst_m2d_9_while_identity_1#
conv_lst_m2d_9_while_identity_2#
conv_lst_m2d_9_while_identity_3#
conv_lst_m2d_9_while_identity_4#
conv_lst_m2d_9_while_identity_55
1conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slices
oconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensorM
2conv_lst_m2d_9_while_split_readvariableop_resource:	?O
4conv_lst_m2d_9_while_split_1_readvariableop_resource:@?C
4conv_lst_m2d_9_while_split_2_readvariableop_resource:	???)conv_lst_m2d_9/while/split/ReadVariableOp?+conv_lst_m2d_9/while/split_1/ReadVariableOp?+conv_lst_m2d_9/while/split_2/ReadVariableOp?
Fconv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   2H
Fconv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0 conv_lst_m2d_9_while_placeholderOconv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02:
8conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem?
$conv_lst_m2d_9/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_9/while/split/split_dim?
)conv_lst_m2d_9/while/split/ReadVariableOpReadVariableOp4conv_lst_m2d_9_while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02+
)conv_lst_m2d_9/while/split/ReadVariableOp?
conv_lst_m2d_9/while/splitSplit-conv_lst_m2d_9/while/split/split_dim:output:01conv_lst_m2d_9/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
conv_lst_m2d_9/while/split?
&conv_lst_m2d_9/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&conv_lst_m2d_9/while/split_1/split_dim?
+conv_lst_m2d_9/while/split_1/ReadVariableOpReadVariableOp6conv_lst_m2d_9_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02-
+conv_lst_m2d_9/while/split_1/ReadVariableOp?
conv_lst_m2d_9/while/split_1Split/conv_lst_m2d_9/while/split_1/split_dim:output:03conv_lst_m2d_9/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_9/while/split_1?
&conv_lst_m2d_9/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&conv_lst_m2d_9/while/split_2/split_dim?
+conv_lst_m2d_9/while/split_2/ReadVariableOpReadVariableOp6conv_lst_m2d_9_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02-
+conv_lst_m2d_9/while/split_2/ReadVariableOp?
conv_lst_m2d_9/while/split_2Split/conv_lst_m2d_9/while/split_2/split_dim:output:03conv_lst_m2d_9/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_9/while/split_2?
 conv_lst_m2d_9/while/convolutionConv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2"
 conv_lst_m2d_9/while/convolution?
conv_lst_m2d_9/while/BiasAddBiasAdd)conv_lst_m2d_9/while/convolution:output:0%conv_lst_m2d_9/while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/BiasAdd?
"conv_lst_m2d_9/while/convolution_1Conv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2$
"conv_lst_m2d_9/while/convolution_1?
conv_lst_m2d_9/while/BiasAdd_1BiasAdd+conv_lst_m2d_9/while/convolution_1:output:0%conv_lst_m2d_9/while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/while/BiasAdd_1?
"conv_lst_m2d_9/while/convolution_2Conv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2$
"conv_lst_m2d_9/while/convolution_2?
conv_lst_m2d_9/while/BiasAdd_2BiasAdd+conv_lst_m2d_9/while/convolution_2:output:0%conv_lst_m2d_9/while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/while/BiasAdd_2?
"conv_lst_m2d_9/while/convolution_3Conv2D?conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:0#conv_lst_m2d_9/while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2$
"conv_lst_m2d_9/while/convolution_3?
conv_lst_m2d_9/while/BiasAdd_3BiasAdd+conv_lst_m2d_9/while/convolution_3:output:0%conv_lst_m2d_9/while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/while/BiasAdd_3?
"conv_lst_m2d_9/while/convolution_4Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_4?
"conv_lst_m2d_9/while/convolution_5Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_5?
"conv_lst_m2d_9/while/convolution_6Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_6?
"conv_lst_m2d_9/while/convolution_7Conv2D"conv_lst_m2d_9_while_placeholder_2%conv_lst_m2d_9/while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2$
"conv_lst_m2d_9/while/convolution_7?
conv_lst_m2d_9/while/addAddV2%conv_lst_m2d_9/while/BiasAdd:output:0+conv_lst_m2d_9/while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add}
conv_lst_m2d_9/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/while/Const?
conv_lst_m2d_9/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/while/Const_1?
conv_lst_m2d_9/while/MulMulconv_lst_m2d_9/while/add:z:0#conv_lst_m2d_9/while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Mul?
conv_lst_m2d_9/while/Add_1Addconv_lst_m2d_9/while/Mul:z:0%conv_lst_m2d_9/while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Add_1?
,conv_lst_m2d_9/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,conv_lst_m2d_9/while/clip_by_value/Minimum/y?
*conv_lst_m2d_9/while/clip_by_value/MinimumMinimumconv_lst_m2d_9/while/Add_1:z:05conv_lst_m2d_9/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2,
*conv_lst_m2d_9/while/clip_by_value/Minimum?
$conv_lst_m2d_9/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv_lst_m2d_9/while/clip_by_value/y?
"conv_lst_m2d_9/while/clip_by_valueMaximum.conv_lst_m2d_9/while/clip_by_value/Minimum:z:0-conv_lst_m2d_9/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2$
"conv_lst_m2d_9/while/clip_by_value?
conv_lst_m2d_9/while/add_2AddV2'conv_lst_m2d_9/while/BiasAdd_1:output:0+conv_lst_m2d_9/while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_2?
conv_lst_m2d_9/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/while/Const_2?
conv_lst_m2d_9/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/while/Const_3?
conv_lst_m2d_9/while/Mul_1Mulconv_lst_m2d_9/while/add_2:z:0%conv_lst_m2d_9/while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Mul_1?
conv_lst_m2d_9/while/Add_3Addconv_lst_m2d_9/while/Mul_1:z:0%conv_lst_m2d_9/while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Add_3?
.conv_lst_m2d_9/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_9/while/clip_by_value_1/Minimum/y?
,conv_lst_m2d_9/while/clip_by_value_1/MinimumMinimumconv_lst_m2d_9/while/Add_3:z:07conv_lst_m2d_9/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2.
,conv_lst_m2d_9/while/clip_by_value_1/Minimum?
&conv_lst_m2d_9/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_9/while/clip_by_value_1/y?
$conv_lst_m2d_9/while/clip_by_value_1Maximum0conv_lst_m2d_9/while/clip_by_value_1/Minimum:z:0/conv_lst_m2d_9/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2&
$conv_lst_m2d_9/while/clip_by_value_1?
conv_lst_m2d_9/while/mul_2Mul(conv_lst_m2d_9/while/clip_by_value_1:z:0"conv_lst_m2d_9_while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/mul_2?
conv_lst_m2d_9/while/add_4AddV2'conv_lst_m2d_9/while/BiasAdd_2:output:0+conv_lst_m2d_9/while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_4?
conv_lst_m2d_9/while/ReluReluconv_lst_m2d_9/while/add_4:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Relu?
conv_lst_m2d_9/while/mul_3Mul&conv_lst_m2d_9/while/clip_by_value:z:0'conv_lst_m2d_9/while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/mul_3?
conv_lst_m2d_9/while/add_5AddV2conv_lst_m2d_9/while/mul_2:z:0conv_lst_m2d_9/while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_5?
conv_lst_m2d_9/while/add_6AddV2'conv_lst_m2d_9/while/BiasAdd_3:output:0+conv_lst_m2d_9/while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/add_6?
conv_lst_m2d_9/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/while/Const_4?
conv_lst_m2d_9/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/while/Const_5?
conv_lst_m2d_9/while/Mul_4Mulconv_lst_m2d_9/while/add_6:z:0%conv_lst_m2d_9/while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Mul_4?
conv_lst_m2d_9/while/Add_7Addconv_lst_m2d_9/while/Mul_4:z:0%conv_lst_m2d_9/while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Add_7?
.conv_lst_m2d_9/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.conv_lst_m2d_9/while/clip_by_value_2/Minimum/y?
,conv_lst_m2d_9/while/clip_by_value_2/MinimumMinimumconv_lst_m2d_9/while/Add_7:z:07conv_lst_m2d_9/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2.
,conv_lst_m2d_9/while/clip_by_value_2/Minimum?
&conv_lst_m2d_9/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&conv_lst_m2d_9/while/clip_by_value_2/y?
$conv_lst_m2d_9/while/clip_by_value_2Maximum0conv_lst_m2d_9/while/clip_by_value_2/Minimum:z:0/conv_lst_m2d_9/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2&
$conv_lst_m2d_9/while/clip_by_value_2?
conv_lst_m2d_9/while/Relu_1Reluconv_lst_m2d_9/while/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/Relu_1?
conv_lst_m2d_9/while/mul_5Mul(conv_lst_m2d_9/while/clip_by_value_2:z:0)conv_lst_m2d_9/while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/while/mul_5?
9conv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"conv_lst_m2d_9_while_placeholder_1 conv_lst_m2d_9_while_placeholderconv_lst_m2d_9/while/mul_5:z:0*
_output_shapes
: *
element_dtype02;
9conv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItem~
conv_lst_m2d_9/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_9/while/add_8/y?
conv_lst_m2d_9/while/add_8AddV2 conv_lst_m2d_9_while_placeholder%conv_lst_m2d_9/while/add_8/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/add_8~
conv_lst_m2d_9/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv_lst_m2d_9/while/add_9/y?
conv_lst_m2d_9/while/add_9AddV26conv_lst_m2d_9_while_conv_lst_m2d_9_while_loop_counter%conv_lst_m2d_9/while/add_9/y:output:0*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/add_9?
conv_lst_m2d_9/while/IdentityIdentityconv_lst_m2d_9/while/add_9:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/Identity?
conv_lst_m2d_9/while/Identity_1Identity<conv_lst_m2d_9_while_conv_lst_m2d_9_while_maximum_iterations*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_9/while/Identity_1?
conv_lst_m2d_9/while/Identity_2Identityconv_lst_m2d_9/while/add_8:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_9/while/Identity_2?
conv_lst_m2d_9/while/Identity_3IdentityIconv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2!
conv_lst_m2d_9/while/Identity_3?
conv_lst_m2d_9/while/Identity_4Identityconv_lst_m2d_9/while/mul_5:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2!
conv_lst_m2d_9/while/Identity_4?
conv_lst_m2d_9/while/Identity_5Identityconv_lst_m2d_9/while/add_5:z:0*^conv_lst_m2d_9/while/split/ReadVariableOp,^conv_lst_m2d_9/while/split_1/ReadVariableOp,^conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2!
conv_lst_m2d_9/while/Identity_5"h
1conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slice3conv_lst_m2d_9_while_conv_lst_m2d_9_strided_slice_0"G
conv_lst_m2d_9_while_identity&conv_lst_m2d_9/while/Identity:output:0"K
conv_lst_m2d_9_while_identity_1(conv_lst_m2d_9/while/Identity_1:output:0"K
conv_lst_m2d_9_while_identity_2(conv_lst_m2d_9/while/Identity_2:output:0"K
conv_lst_m2d_9_while_identity_3(conv_lst_m2d_9/while/Identity_3:output:0"K
conv_lst_m2d_9_while_identity_4(conv_lst_m2d_9/while/Identity_4:output:0"K
conv_lst_m2d_9_while_identity_5(conv_lst_m2d_9/while/Identity_5:output:0"n
4conv_lst_m2d_9_while_split_1_readvariableop_resource6conv_lst_m2d_9_while_split_1_readvariableop_resource_0"n
4conv_lst_m2d_9_while_split_2_readvariableop_resource6conv_lst_m2d_9_while_split_2_readvariableop_resource_0"j
2conv_lst_m2d_9_while_split_readvariableop_resource4conv_lst_m2d_9_while_split_readvariableop_resource_0"?
oconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensorqconv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 2V
)conv_lst_m2d_9/while/split/ReadVariableOp)conv_lst_m2d_9/while/split/ReadVariableOp2Z
+conv_lst_m2d_9/while/split_1/ReadVariableOp+conv_lst_m2d_9/while/split_1/ReadVariableOp2Z
+conv_lst_m2d_9/while/split_2/ReadVariableOp+conv_lst_m2d_9/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?
`
D__inference_flatten_9_layer_call_and_return_conditional_losses_63502

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????12	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
?p
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_62798
inputs_08
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&??????????????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_8w
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62672*
condR
while_cond_62671*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&??????????????????d	: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
<
_output_shapes*
(:&??????????????????d	
"
_user_specified_name
inputs/0
?D
?
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_60896

inputs

states
states_18
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7w
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
IdentityIdentity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity?

Identity_1Identity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_1?

Identity_2Identity	add_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????d	:?????????b@:?????????b@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:?????????d	
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????b@
 
_user_specified_namestates:WS
/
_output_shapes
:?????????b@
 
_user_specified_namestates
?	
?
#__inference_signature_wrapper_62009
conv_lst_m2d_9_input"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?1d
	unknown_3:d
	unknown_4:d
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv_lst_m2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_606042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
3
_output_shapes!
:?????????d	
.
_user_specified_nameconv_lst_m2d_9_input
ά
?
,sequential_9_conv_lst_m2d_9_while_body_60461T
Psequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_loop_counterZ
Vsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_maximum_iterations1
-sequential_9_conv_lst_m2d_9_while_placeholder3
/sequential_9_conv_lst_m2d_9_while_placeholder_13
/sequential_9_conv_lst_m2d_9_while_placeholder_23
/sequential_9_conv_lst_m2d_9_while_placeholder_3Q
Msequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_strided_slice_0?
?sequential_9_conv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0\
Asequential_9_conv_lst_m2d_9_while_split_readvariableop_resource_0:	?^
Csequential_9_conv_lst_m2d_9_while_split_1_readvariableop_resource_0:@?R
Csequential_9_conv_lst_m2d_9_while_split_2_readvariableop_resource_0:	?.
*sequential_9_conv_lst_m2d_9_while_identity0
,sequential_9_conv_lst_m2d_9_while_identity_10
,sequential_9_conv_lst_m2d_9_while_identity_20
,sequential_9_conv_lst_m2d_9_while_identity_30
,sequential_9_conv_lst_m2d_9_while_identity_40
,sequential_9_conv_lst_m2d_9_while_identity_5O
Ksequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_strided_slice?
?sequential_9_conv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensorZ
?sequential_9_conv_lst_m2d_9_while_split_readvariableop_resource:	?\
Asequential_9_conv_lst_m2d_9_while_split_1_readvariableop_resource:@?P
Asequential_9_conv_lst_m2d_9_while_split_2_readvariableop_resource:	???6sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp?8sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp?8sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp?
Ssequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   2U
Ssequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Esequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_9_conv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0-sequential_9_conv_lst_m2d_9_while_placeholder\sequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02G
Esequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem?
1sequential_9/conv_lst_m2d_9/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/conv_lst_m2d_9/while/split/split_dim?
6sequential_9/conv_lst_m2d_9/while/split/ReadVariableOpReadVariableOpAsequential_9_conv_lst_m2d_9_while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype028
6sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp?
'sequential_9/conv_lst_m2d_9/while/splitSplit:sequential_9/conv_lst_m2d_9/while/split/split_dim:output:0>sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2)
'sequential_9/conv_lst_m2d_9/while/split?
3sequential_9/conv_lst_m2d_9/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3sequential_9/conv_lst_m2d_9/while/split_1/split_dim?
8sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOpReadVariableOpCsequential_9_conv_lst_m2d_9_while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02:
8sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp?
)sequential_9/conv_lst_m2d_9/while/split_1Split<sequential_9/conv_lst_m2d_9/while/split_1/split_dim:output:0@sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2+
)sequential_9/conv_lst_m2d_9/while/split_1?
3sequential_9/conv_lst_m2d_9/while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_9/conv_lst_m2d_9/while/split_2/split_dim?
8sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOpReadVariableOpCsequential_9_conv_lst_m2d_9_while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02:
8sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp?
)sequential_9/conv_lst_m2d_9/while/split_2Split<sequential_9/conv_lst_m2d_9/while/split_2/split_dim:output:0@sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2+
)sequential_9/conv_lst_m2d_9/while/split_2?
-sequential_9/conv_lst_m2d_9/while/convolutionConv2DLsequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:00sequential_9/conv_lst_m2d_9/while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2/
-sequential_9/conv_lst_m2d_9/while/convolution?
)sequential_9/conv_lst_m2d_9/while/BiasAddBiasAdd6sequential_9/conv_lst_m2d_9/while/convolution:output:02sequential_9/conv_lst_m2d_9/while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2+
)sequential_9/conv_lst_m2d_9/while/BiasAdd?
/sequential_9/conv_lst_m2d_9/while/convolution_1Conv2DLsequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:00sequential_9/conv_lst_m2d_9/while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
21
/sequential_9/conv_lst_m2d_9/while/convolution_1?
+sequential_9/conv_lst_m2d_9/while/BiasAdd_1BiasAdd8sequential_9/conv_lst_m2d_9/while/convolution_1:output:02sequential_9/conv_lst_m2d_9/while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2-
+sequential_9/conv_lst_m2d_9/while/BiasAdd_1?
/sequential_9/conv_lst_m2d_9/while/convolution_2Conv2DLsequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:00sequential_9/conv_lst_m2d_9/while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
21
/sequential_9/conv_lst_m2d_9/while/convolution_2?
+sequential_9/conv_lst_m2d_9/while/BiasAdd_2BiasAdd8sequential_9/conv_lst_m2d_9/while/convolution_2:output:02sequential_9/conv_lst_m2d_9/while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2-
+sequential_9/conv_lst_m2d_9/while/BiasAdd_2?
/sequential_9/conv_lst_m2d_9/while/convolution_3Conv2DLsequential_9/conv_lst_m2d_9/while/TensorArrayV2Read/TensorListGetItem:item:00sequential_9/conv_lst_m2d_9/while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
21
/sequential_9/conv_lst_m2d_9/while/convolution_3?
+sequential_9/conv_lst_m2d_9/while/BiasAdd_3BiasAdd8sequential_9/conv_lst_m2d_9/while/convolution_3:output:02sequential_9/conv_lst_m2d_9/while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2-
+sequential_9/conv_lst_m2d_9/while/BiasAdd_3?
/sequential_9/conv_lst_m2d_9/while/convolution_4Conv2D/sequential_9_conv_lst_m2d_9_while_placeholder_22sequential_9/conv_lst_m2d_9/while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
21
/sequential_9/conv_lst_m2d_9/while/convolution_4?
/sequential_9/conv_lst_m2d_9/while/convolution_5Conv2D/sequential_9_conv_lst_m2d_9_while_placeholder_22sequential_9/conv_lst_m2d_9/while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
21
/sequential_9/conv_lst_m2d_9/while/convolution_5?
/sequential_9/conv_lst_m2d_9/while/convolution_6Conv2D/sequential_9_conv_lst_m2d_9_while_placeholder_22sequential_9/conv_lst_m2d_9/while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
21
/sequential_9/conv_lst_m2d_9/while/convolution_6?
/sequential_9/conv_lst_m2d_9/while/convolution_7Conv2D/sequential_9_conv_lst_m2d_9_while_placeholder_22sequential_9/conv_lst_m2d_9/while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
21
/sequential_9/conv_lst_m2d_9/while/convolution_7?
%sequential_9/conv_lst_m2d_9/while/addAddV22sequential_9/conv_lst_m2d_9/while/BiasAdd:output:08sequential_9/conv_lst_m2d_9/while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2'
%sequential_9/conv_lst_m2d_9/while/add?
'sequential_9/conv_lst_m2d_9/while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'sequential_9/conv_lst_m2d_9/while/Const?
)sequential_9/conv_lst_m2d_9/while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)sequential_9/conv_lst_m2d_9/while/Const_1?
%sequential_9/conv_lst_m2d_9/while/MulMul)sequential_9/conv_lst_m2d_9/while/add:z:00sequential_9/conv_lst_m2d_9/while/Const:output:0*
T0*/
_output_shapes
:?????????b@2'
%sequential_9/conv_lst_m2d_9/while/Mul?
'sequential_9/conv_lst_m2d_9/while/Add_1Add)sequential_9/conv_lst_m2d_9/while/Mul:z:02sequential_9/conv_lst_m2d_9/while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/Add_1?
9sequential_9/conv_lst_m2d_9/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2;
9sequential_9/conv_lst_m2d_9/while/clip_by_value/Minimum/y?
7sequential_9/conv_lst_m2d_9/while/clip_by_value/MinimumMinimum+sequential_9/conv_lst_m2d_9/while/Add_1:z:0Bsequential_9/conv_lst_m2d_9/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@29
7sequential_9/conv_lst_m2d_9/while/clip_by_value/Minimum?
1sequential_9/conv_lst_m2d_9/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential_9/conv_lst_m2d_9/while/clip_by_value/y?
/sequential_9/conv_lst_m2d_9/while/clip_by_valueMaximum;sequential_9/conv_lst_m2d_9/while/clip_by_value/Minimum:z:0:sequential_9/conv_lst_m2d_9/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@21
/sequential_9/conv_lst_m2d_9/while/clip_by_value?
'sequential_9/conv_lst_m2d_9/while/add_2AddV24sequential_9/conv_lst_m2d_9/while/BiasAdd_1:output:08sequential_9/conv_lst_m2d_9/while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/add_2?
)sequential_9/conv_lst_m2d_9/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)sequential_9/conv_lst_m2d_9/while/Const_2?
)sequential_9/conv_lst_m2d_9/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)sequential_9/conv_lst_m2d_9/while/Const_3?
'sequential_9/conv_lst_m2d_9/while/Mul_1Mul+sequential_9/conv_lst_m2d_9/while/add_2:z:02sequential_9/conv_lst_m2d_9/while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/Mul_1?
'sequential_9/conv_lst_m2d_9/while/Add_3Add+sequential_9/conv_lst_m2d_9/while/Mul_1:z:02sequential_9/conv_lst_m2d_9/while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/Add_3?
;sequential_9/conv_lst_m2d_9/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2=
;sequential_9/conv_lst_m2d_9/while/clip_by_value_1/Minimum/y?
9sequential_9/conv_lst_m2d_9/while/clip_by_value_1/MinimumMinimum+sequential_9/conv_lst_m2d_9/while/Add_3:z:0Dsequential_9/conv_lst_m2d_9/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2;
9sequential_9/conv_lst_m2d_9/while/clip_by_value_1/Minimum?
3sequential_9/conv_lst_m2d_9/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3sequential_9/conv_lst_m2d_9/while/clip_by_value_1/y?
1sequential_9/conv_lst_m2d_9/while/clip_by_value_1Maximum=sequential_9/conv_lst_m2d_9/while/clip_by_value_1/Minimum:z:0<sequential_9/conv_lst_m2d_9/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@23
1sequential_9/conv_lst_m2d_9/while/clip_by_value_1?
'sequential_9/conv_lst_m2d_9/while/mul_2Mul5sequential_9/conv_lst_m2d_9/while/clip_by_value_1:z:0/sequential_9_conv_lst_m2d_9_while_placeholder_3*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/mul_2?
'sequential_9/conv_lst_m2d_9/while/add_4AddV24sequential_9/conv_lst_m2d_9/while/BiasAdd_2:output:08sequential_9/conv_lst_m2d_9/while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/add_4?
&sequential_9/conv_lst_m2d_9/while/ReluRelu+sequential_9/conv_lst_m2d_9/while/add_4:z:0*
T0*/
_output_shapes
:?????????b@2(
&sequential_9/conv_lst_m2d_9/while/Relu?
'sequential_9/conv_lst_m2d_9/while/mul_3Mul3sequential_9/conv_lst_m2d_9/while/clip_by_value:z:04sequential_9/conv_lst_m2d_9/while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/mul_3?
'sequential_9/conv_lst_m2d_9/while/add_5AddV2+sequential_9/conv_lst_m2d_9/while/mul_2:z:0+sequential_9/conv_lst_m2d_9/while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/add_5?
'sequential_9/conv_lst_m2d_9/while/add_6AddV24sequential_9/conv_lst_m2d_9/while/BiasAdd_3:output:08sequential_9/conv_lst_m2d_9/while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/add_6?
)sequential_9/conv_lst_m2d_9/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2+
)sequential_9/conv_lst_m2d_9/while/Const_4?
)sequential_9/conv_lst_m2d_9/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)sequential_9/conv_lst_m2d_9/while/Const_5?
'sequential_9/conv_lst_m2d_9/while/Mul_4Mul+sequential_9/conv_lst_m2d_9/while/add_6:z:02sequential_9/conv_lst_m2d_9/while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/Mul_4?
'sequential_9/conv_lst_m2d_9/while/Add_7Add+sequential_9/conv_lst_m2d_9/while/Mul_4:z:02sequential_9/conv_lst_m2d_9/while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/Add_7?
;sequential_9/conv_lst_m2d_9/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2=
;sequential_9/conv_lst_m2d_9/while/clip_by_value_2/Minimum/y?
9sequential_9/conv_lst_m2d_9/while/clip_by_value_2/MinimumMinimum+sequential_9/conv_lst_m2d_9/while/Add_7:z:0Dsequential_9/conv_lst_m2d_9/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2;
9sequential_9/conv_lst_m2d_9/while/clip_by_value_2/Minimum?
3sequential_9/conv_lst_m2d_9/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3sequential_9/conv_lst_m2d_9/while/clip_by_value_2/y?
1sequential_9/conv_lst_m2d_9/while/clip_by_value_2Maximum=sequential_9/conv_lst_m2d_9/while/clip_by_value_2/Minimum:z:0<sequential_9/conv_lst_m2d_9/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@23
1sequential_9/conv_lst_m2d_9/while/clip_by_value_2?
(sequential_9/conv_lst_m2d_9/while/Relu_1Relu+sequential_9/conv_lst_m2d_9/while/add_5:z:0*
T0*/
_output_shapes
:?????????b@2*
(sequential_9/conv_lst_m2d_9/while/Relu_1?
'sequential_9/conv_lst_m2d_9/while/mul_5Mul5sequential_9/conv_lst_m2d_9/while/clip_by_value_2:z:06sequential_9/conv_lst_m2d_9/while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2)
'sequential_9/conv_lst_m2d_9/while/mul_5?
Fsequential_9/conv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem/sequential_9_conv_lst_m2d_9_while_placeholder_1-sequential_9_conv_lst_m2d_9_while_placeholder+sequential_9/conv_lst_m2d_9/while/mul_5:z:0*
_output_shapes
: *
element_dtype02H
Fsequential_9/conv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItem?
)sequential_9/conv_lst_m2d_9/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_9/conv_lst_m2d_9/while/add_8/y?
'sequential_9/conv_lst_m2d_9/while/add_8AddV2-sequential_9_conv_lst_m2d_9_while_placeholder2sequential_9/conv_lst_m2d_9/while/add_8/y:output:0*
T0*
_output_shapes
: 2)
'sequential_9/conv_lst_m2d_9/while/add_8?
)sequential_9/conv_lst_m2d_9/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_9/conv_lst_m2d_9/while/add_9/y?
'sequential_9/conv_lst_m2d_9/while/add_9AddV2Psequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_loop_counter2sequential_9/conv_lst_m2d_9/while/add_9/y:output:0*
T0*
_output_shapes
: 2)
'sequential_9/conv_lst_m2d_9/while/add_9?
*sequential_9/conv_lst_m2d_9/while/IdentityIdentity+sequential_9/conv_lst_m2d_9/while/add_9:z:07^sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2,
*sequential_9/conv_lst_m2d_9/while/Identity?
,sequential_9/conv_lst_m2d_9/while/Identity_1IdentityVsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_maximum_iterations7^sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_9/conv_lst_m2d_9/while/Identity_1?
,sequential_9/conv_lst_m2d_9/while/Identity_2Identity+sequential_9/conv_lst_m2d_9/while/add_8:z:07^sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_9/conv_lst_m2d_9/while/Identity_2?
,sequential_9/conv_lst_m2d_9/while/Identity_3IdentityVsequential_9/conv_lst_m2d_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2.
,sequential_9/conv_lst_m2d_9/while/Identity_3?
,sequential_9/conv_lst_m2d_9/while/Identity_4Identity+sequential_9/conv_lst_m2d_9/while/mul_5:z:07^sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2.
,sequential_9/conv_lst_m2d_9/while/Identity_4?
,sequential_9/conv_lst_m2d_9/while/Identity_5Identity+sequential_9/conv_lst_m2d_9/while/add_5:z:07^sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp9^sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2.
,sequential_9/conv_lst_m2d_9/while/Identity_5"a
*sequential_9_conv_lst_m2d_9_while_identity3sequential_9/conv_lst_m2d_9/while/Identity:output:0"e
,sequential_9_conv_lst_m2d_9_while_identity_15sequential_9/conv_lst_m2d_9/while/Identity_1:output:0"e
,sequential_9_conv_lst_m2d_9_while_identity_25sequential_9/conv_lst_m2d_9/while/Identity_2:output:0"e
,sequential_9_conv_lst_m2d_9_while_identity_35sequential_9/conv_lst_m2d_9/while/Identity_3:output:0"e
,sequential_9_conv_lst_m2d_9_while_identity_45sequential_9/conv_lst_m2d_9/while/Identity_4:output:0"e
,sequential_9_conv_lst_m2d_9_while_identity_55sequential_9/conv_lst_m2d_9/while/Identity_5:output:0"?
Ksequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_strided_sliceMsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_strided_slice_0"?
Asequential_9_conv_lst_m2d_9_while_split_1_readvariableop_resourceCsequential_9_conv_lst_m2d_9_while_split_1_readvariableop_resource_0"?
Asequential_9_conv_lst_m2d_9_while_split_2_readvariableop_resourceCsequential_9_conv_lst_m2d_9_while_split_2_readvariableop_resource_0"?
?sequential_9_conv_lst_m2d_9_while_split_readvariableop_resourceAsequential_9_conv_lst_m2d_9_while_split_readvariableop_resource_0"?
?sequential_9_conv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor?sequential_9_conv_lst_m2d_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_conv_lst_m2d_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 2p
6sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp6sequential_9/conv_lst_m2d_9/while/split/ReadVariableOp2t
8sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp8sequential_9/conv_lst_m2d_9/while/split_1/ReadVariableOp2t
8sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp8sequential_9/conv_lst_m2d_9/while/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?
?
conv_lst_m2d_9_while_cond_62381:
6conv_lst_m2d_9_while_conv_lst_m2d_9_while_loop_counter@
<conv_lst_m2d_9_while_conv_lst_m2d_9_while_maximum_iterations$
 conv_lst_m2d_9_while_placeholder&
"conv_lst_m2d_9_while_placeholder_1&
"conv_lst_m2d_9_while_placeholder_2&
"conv_lst_m2d_9_while_placeholder_3:
6conv_lst_m2d_9_while_less_conv_lst_m2d_9_strided_sliceQ
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62381___redundant_placeholder0Q
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62381___redundant_placeholder1Q
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62381___redundant_placeholder2Q
Mconv_lst_m2d_9_while_conv_lst_m2d_9_while_cond_62381___redundant_placeholder3!
conv_lst_m2d_9_while_identity
?
conv_lst_m2d_9/while/LessLess conv_lst_m2d_9_while_placeholder6conv_lst_m2d_9_while_less_conv_lst_m2d_9_strided_slice*
T0*
_output_shapes
: 2
conv_lst_m2d_9/while/Less?
conv_lst_m2d_9/while/IdentityIdentityconv_lst_m2d_9/while/Less:z:0*
T0
*
_output_shapes
: 2
conv_lst_m2d_9/while/Identity"G
conv_lst_m2d_9_while_identity&conv_lst_m2d_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?h
?
while_body_62672
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:	?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:	?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1|
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_4k

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:?????????b@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2o
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9?
while/IdentityIdentitywhile/add_9:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add_8:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?h
?
while_body_62894
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:	?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:	?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1|
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_4k

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:?????????b@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2o
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9?
while/IdentityIdentitywhile/add_9:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add_8:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?	
?
,__inference_sequential_9_layer_call_fn_62028

inputs"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?1d
	unknown_3:d
	unknown_4:d
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_615462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
`
D__inference_flatten_9_layer_call_and_return_conditional_losses_61509

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????12	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
?	
?
,__inference_sequential_9_layer_call_fn_62047

inputs"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?1d
	unknown_3:d
	unknown_4:d
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_619002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_61501

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????b@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????b@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
??
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_62286

inputsG
,conv_lst_m2d_9_split_readvariableop_resource:	?I
.conv_lst_m2d_9_split_1_readvariableop_resource:@?=
.conv_lst_m2d_9_split_2_readvariableop_resource:	?:
'dense_18_matmul_readvariableop_resource:	?1d6
(dense_18_biasadd_readvariableop_resource:d9
'dense_19_matmul_readvariableop_resource:d6
(dense_19_biasadd_readvariableop_resource:
identity??#conv_lst_m2d_9/split/ReadVariableOp?%conv_lst_m2d_9/split_1/ReadVariableOp?%conv_lst_m2d_9/split_2/ReadVariableOp?conv_lst_m2d_9/while?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
conv_lst_m2d_9/zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????d	2
conv_lst_m2d_9/zeros_like?
$conv_lst_m2d_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$conv_lst_m2d_9/Sum/reduction_indices?
conv_lst_m2d_9/SumSumconv_lst_m2d_9/zeros_like:y:0-conv_lst_m2d_9/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
conv_lst_m2d_9/Sum?
$conv_lst_m2d_9/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2&
$conv_lst_m2d_9/zeros/shape_as_tensor}
conv_lst_m2d_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
conv_lst_m2d_9/zeros/Const?
conv_lst_m2d_9/zerosFill-conv_lst_m2d_9/zeros/shape_as_tensor:output:0#conv_lst_m2d_9/zeros/Const:output:0*
T0*&
_output_shapes
:	@2
conv_lst_m2d_9/zeros?
conv_lst_m2d_9/convolutionConv2Dconv_lst_m2d_9/Sum:output:0conv_lst_m2d_9/zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution?
conv_lst_m2d_9/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
conv_lst_m2d_9/transpose/perm?
conv_lst_m2d_9/transpose	Transposeinputs&conv_lst_m2d_9/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????d	2
conv_lst_m2d_9/transposex
conv_lst_m2d_9/ShapeShapeconv_lst_m2d_9/transpose:y:0*
T0*
_output_shapes
:2
conv_lst_m2d_9/Shape?
"conv_lst_m2d_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"conv_lst_m2d_9/strided_slice/stack?
$conv_lst_m2d_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_9/strided_slice/stack_1?
$conv_lst_m2d_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$conv_lst_m2d_9/strided_slice/stack_2?
conv_lst_m2d_9/strided_sliceStridedSliceconv_lst_m2d_9/Shape:output:0+conv_lst_m2d_9/strided_slice/stack:output:0-conv_lst_m2d_9/strided_slice/stack_1:output:0-conv_lst_m2d_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
conv_lst_m2d_9/strided_slice?
*conv_lst_m2d_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*conv_lst_m2d_9/TensorArrayV2/element_shape?
conv_lst_m2d_9/TensorArrayV2TensorListReserve3conv_lst_m2d_9/TensorArrayV2/element_shape:output:0%conv_lst_m2d_9/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
conv_lst_m2d_9/TensorArrayV2?
Dconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   2F
Dconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shape?
6conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconv_lst_m2d_9/transpose:y:0Mconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6conv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor?
$conv_lst_m2d_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_lst_m2d_9/strided_slice_1/stack?
&conv_lst_m2d_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_9/strided_slice_1/stack_1?
&conv_lst_m2d_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_9/strided_slice_1/stack_2?
conv_lst_m2d_9/strided_slice_1StridedSliceconv_lst_m2d_9/transpose:y:0-conv_lst_m2d_9/strided_slice_1/stack:output:0/conv_lst_m2d_9/strided_slice_1/stack_1:output:0/conv_lst_m2d_9/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2 
conv_lst_m2d_9/strided_slice_1?
conv_lst_m2d_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
conv_lst_m2d_9/split/split_dim?
#conv_lst_m2d_9/split/ReadVariableOpReadVariableOp,conv_lst_m2d_9_split_readvariableop_resource*'
_output_shapes
:	?*
dtype02%
#conv_lst_m2d_9/split/ReadVariableOp?
conv_lst_m2d_9/splitSplit'conv_lst_m2d_9/split/split_dim:output:0+conv_lst_m2d_9/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
conv_lst_m2d_9/split?
 conv_lst_m2d_9/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 conv_lst_m2d_9/split_1/split_dim?
%conv_lst_m2d_9/split_1/ReadVariableOpReadVariableOp.conv_lst_m2d_9_split_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02'
%conv_lst_m2d_9/split_1/ReadVariableOp?
conv_lst_m2d_9/split_1Split)conv_lst_m2d_9/split_1/split_dim:output:0-conv_lst_m2d_9/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
conv_lst_m2d_9/split_1?
 conv_lst_m2d_9/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv_lst_m2d_9/split_2/split_dim?
%conv_lst_m2d_9/split_2/ReadVariableOpReadVariableOp.conv_lst_m2d_9_split_2_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%conv_lst_m2d_9/split_2/ReadVariableOp?
conv_lst_m2d_9/split_2Split)conv_lst_m2d_9/split_2/split_dim:output:0-conv_lst_m2d_9/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
conv_lst_m2d_9/split_2?
conv_lst_m2d_9/convolution_1Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_1?
conv_lst_m2d_9/BiasAddBiasAdd%conv_lst_m2d_9/convolution_1:output:0conv_lst_m2d_9/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd?
conv_lst_m2d_9/convolution_2Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_2?
conv_lst_m2d_9/BiasAdd_1BiasAdd%conv_lst_m2d_9/convolution_2:output:0conv_lst_m2d_9/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd_1?
conv_lst_m2d_9/convolution_3Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_3?
conv_lst_m2d_9/BiasAdd_2BiasAdd%conv_lst_m2d_9/convolution_3:output:0conv_lst_m2d_9/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd_2?
conv_lst_m2d_9/convolution_4Conv2D'conv_lst_m2d_9/strided_slice_1:output:0conv_lst_m2d_9/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
conv_lst_m2d_9/convolution_4?
conv_lst_m2d_9/BiasAdd_3BiasAdd%conv_lst_m2d_9/convolution_4:output:0conv_lst_m2d_9/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/BiasAdd_3?
conv_lst_m2d_9/convolution_5Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_5?
conv_lst_m2d_9/convolution_6Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_6?
conv_lst_m2d_9/convolution_7Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_7?
conv_lst_m2d_9/convolution_8Conv2D#conv_lst_m2d_9/convolution:output:0conv_lst_m2d_9/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
conv_lst_m2d_9/convolution_8?
conv_lst_m2d_9/addAddV2conv_lst_m2d_9/BiasAdd:output:0%conv_lst_m2d_9/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/addq
conv_lst_m2d_9/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/Constu
conv_lst_m2d_9/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/Const_1?
conv_lst_m2d_9/MulMulconv_lst_m2d_9/add:z:0conv_lst_m2d_9/Const:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Mul?
conv_lst_m2d_9/Add_1Addconv_lst_m2d_9/Mul:z:0conv_lst_m2d_9/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Add_1?
&conv_lst_m2d_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&conv_lst_m2d_9/clip_by_value/Minimum/y?
$conv_lst_m2d_9/clip_by_value/MinimumMinimumconv_lst_m2d_9/Add_1:z:0/conv_lst_m2d_9/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2&
$conv_lst_m2d_9/clip_by_value/Minimum?
conv_lst_m2d_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
conv_lst_m2d_9/clip_by_value/y?
conv_lst_m2d_9/clip_by_valueMaximum(conv_lst_m2d_9/clip_by_value/Minimum:z:0'conv_lst_m2d_9/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/clip_by_value?
conv_lst_m2d_9/add_2AddV2!conv_lst_m2d_9/BiasAdd_1:output:0%conv_lst_m2d_9/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_2u
conv_lst_m2d_9/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/Const_2u
conv_lst_m2d_9/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/Const_3?
conv_lst_m2d_9/Mul_1Mulconv_lst_m2d_9/add_2:z:0conv_lst_m2d_9/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Mul_1?
conv_lst_m2d_9/Add_3Addconv_lst_m2d_9/Mul_1:z:0conv_lst_m2d_9/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Add_3?
(conv_lst_m2d_9/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_9/clip_by_value_1/Minimum/y?
&conv_lst_m2d_9/clip_by_value_1/MinimumMinimumconv_lst_m2d_9/Add_3:z:01conv_lst_m2d_9/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2(
&conv_lst_m2d_9/clip_by_value_1/Minimum?
 conv_lst_m2d_9/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_9/clip_by_value_1/y?
conv_lst_m2d_9/clip_by_value_1Maximum*conv_lst_m2d_9/clip_by_value_1/Minimum:z:0)conv_lst_m2d_9/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/clip_by_value_1?
conv_lst_m2d_9/mul_2Mul"conv_lst_m2d_9/clip_by_value_1:z:0#conv_lst_m2d_9/convolution:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/mul_2?
conv_lst_m2d_9/add_4AddV2!conv_lst_m2d_9/BiasAdd_2:output:0%conv_lst_m2d_9/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_4?
conv_lst_m2d_9/ReluReluconv_lst_m2d_9/add_4:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Relu?
conv_lst_m2d_9/mul_3Mul conv_lst_m2d_9/clip_by_value:z:0!conv_lst_m2d_9/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/mul_3?
conv_lst_m2d_9/add_5AddV2conv_lst_m2d_9/mul_2:z:0conv_lst_m2d_9/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_5?
conv_lst_m2d_9/add_6AddV2!conv_lst_m2d_9/BiasAdd_3:output:0%conv_lst_m2d_9/convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/add_6u
conv_lst_m2d_9/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
conv_lst_m2d_9/Const_4u
conv_lst_m2d_9/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
conv_lst_m2d_9/Const_5?
conv_lst_m2d_9/Mul_4Mulconv_lst_m2d_9/add_6:z:0conv_lst_m2d_9/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Mul_4?
conv_lst_m2d_9/Add_7Addconv_lst_m2d_9/Mul_4:z:0conv_lst_m2d_9/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Add_7?
(conv_lst_m2d_9/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(conv_lst_m2d_9/clip_by_value_2/Minimum/y?
&conv_lst_m2d_9/clip_by_value_2/MinimumMinimumconv_lst_m2d_9/Add_7:z:01conv_lst_m2d_9/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2(
&conv_lst_m2d_9/clip_by_value_2/Minimum?
 conv_lst_m2d_9/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv_lst_m2d_9/clip_by_value_2/y?
conv_lst_m2d_9/clip_by_value_2Maximum*conv_lst_m2d_9/clip_by_value_2/Minimum:z:0)conv_lst_m2d_9/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2 
conv_lst_m2d_9/clip_by_value_2?
conv_lst_m2d_9/Relu_1Reluconv_lst_m2d_9/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/Relu_1?
conv_lst_m2d_9/mul_5Mul"conv_lst_m2d_9/clip_by_value_2:z:0#conv_lst_m2d_9/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
conv_lst_m2d_9/mul_5?
,conv_lst_m2d_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2.
,conv_lst_m2d_9/TensorArrayV2_1/element_shape?
conv_lst_m2d_9/TensorArrayV2_1TensorListReserve5conv_lst_m2d_9/TensorArrayV2_1/element_shape:output:0%conv_lst_m2d_9/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
conv_lst_m2d_9/TensorArrayV2_1l
conv_lst_m2d_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
conv_lst_m2d_9/time?
'conv_lst_m2d_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'conv_lst_m2d_9/while/maximum_iterations?
!conv_lst_m2d_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv_lst_m2d_9/while/loop_counter?
conv_lst_m2d_9/whileWhile*conv_lst_m2d_9/while/loop_counter:output:00conv_lst_m2d_9/while/maximum_iterations:output:0conv_lst_m2d_9/time:output:0'conv_lst_m2d_9/TensorArrayV2_1:handle:0#conv_lst_m2d_9/convolution:output:0#conv_lst_m2d_9/convolution:output:0%conv_lst_m2d_9/strided_slice:output:0Fconv_lst_m2d_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0,conv_lst_m2d_9_split_readvariableop_resource.conv_lst_m2d_9_split_1_readvariableop_resource.conv_lst_m2d_9_split_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
conv_lst_m2d_9_while_body_62143*+
cond#R!
conv_lst_m2d_9_while_cond_62142*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
conv_lst_m2d_9/while?
?conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2A
?conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shape?
1conv_lst_m2d_9/TensorArrayV2Stack/TensorListStackTensorListStackconv_lst_m2d_9/while:output:3Hconv_lst_m2d_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????b@*
element_dtype023
1conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack?
$conv_lst_m2d_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$conv_lst_m2d_9/strided_slice_2/stack?
&conv_lst_m2d_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_lst_m2d_9/strided_slice_2/stack_1?
&conv_lst_m2d_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_lst_m2d_9/strided_slice_2/stack_2?
conv_lst_m2d_9/strided_slice_2StridedSlice:conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack:tensor:0-conv_lst_m2d_9/strided_slice_2/stack:output:0/conv_lst_m2d_9/strided_slice_2/stack_1:output:0/conv_lst_m2d_9/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2 
conv_lst_m2d_9/strided_slice_2?
conv_lst_m2d_9/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2!
conv_lst_m2d_9/transpose_1/perm?
conv_lst_m2d_9/transpose_1	Transpose:conv_lst_m2d_9/TensorArrayV2Stack/TensorListStack:tensor:0(conv_lst_m2d_9/transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????b@2
conv_lst_m2d_9/transpose_1?
dropout_9/IdentityIdentity'conv_lst_m2d_9/strided_slice_2:output:0*
T0*/
_output_shapes
:?????????b@2
dropout_9/Identitys
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_9/Const?
flatten_9/ReshapeReshapedropout_9/Identity:output:0flatten_9/Const:output:0*
T0*(
_output_shapes
:??????????12
flatten_9/Reshape?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	?1d*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMulflatten_9/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_18/Relu?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdd|
dense_19/SoftmaxSoftmaxdense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Softmax?
IdentityIdentitydense_19/Softmax:softmax:0$^conv_lst_m2d_9/split/ReadVariableOp&^conv_lst_m2d_9/split_1/ReadVariableOp&^conv_lst_m2d_9/split_2/ReadVariableOp^conv_lst_m2d_9/while ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 2J
#conv_lst_m2d_9/split/ReadVariableOp#conv_lst_m2d_9/split/ReadVariableOp2N
%conv_lst_m2d_9/split_1/ReadVariableOp%conv_lst_m2d_9/split_1/ReadVariableOp2N
%conv_lst_m2d_9/split_2/ReadVariableOp%conv_lst_m2d_9/split_2/ReadVariableOp2,
conv_lst_m2d_9/whileconv_lst_m2d_9/while2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_61959
conv_lst_m2d_9_input/
conv_lst_m2d_9_61939:	?/
conv_lst_m2d_9_61941:@?#
conv_lst_m2d_9_61943:	?!
dense_18_61948:	?1d
dense_18_61950:d 
dense_19_61953:d
dense_19_61955:
identity??&conv_lst_m2d_9/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
&conv_lst_m2d_9/StatefulPartitionedCallStatefulPartitionedCallconv_lst_m2d_9_inputconv_lst_m2d_9_61939conv_lst_m2d_9_61941conv_lst_m2d_9_61943*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_614882(
&conv_lst_m2d_9/StatefulPartitionedCall?
dropout_9/PartitionedCallPartitionedCall/conv_lst_m2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_615012
dropout_9/PartitionedCall?
flatten_9/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_615092
flatten_9/PartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_18_61948dense_18_61950*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_615222"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_61953dense_19_61955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_615392"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0'^conv_lst_m2d_9/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 2P
&conv_lst_m2d_9/StatefulPartitionedCall&conv_lst_m2d_9/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:i e
3
_output_shapes!
:?????????d	
.
_user_specified_nameconv_lst_m2d_9_input
?
?
while_cond_60721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_60721___redundant_placeholder03
/while_while_cond_60721___redundant_placeholder13
/while_while_cond_60721___redundant_placeholder23
/while_while_cond_60721___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_63491

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????b@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????b@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????b@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????b@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????b@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
?C
?
__inference__traced_save_63839
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_conv_lst_m2d_9_kernel_read_readvariableop>
:savev2_conv_lst_m2d_9_recurrent_kernel_read_readvariableop2
.savev2_conv_lst_m2d_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop;
7savev2_adam_conv_lst_m2d_9_kernel_m_read_readvariableopE
Asavev2_adam_conv_lst_m2d_9_recurrent_kernel_m_read_readvariableop9
5savev2_adam_conv_lst_m2d_9_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop;
7savev2_adam_conv_lst_m2d_9_kernel_v_read_readvariableopE
Asavev2_adam_conv_lst_m2d_9_recurrent_kernel_v_read_readvariableop9
5savev2_adam_conv_lst_m2d_9_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_conv_lst_m2d_9_kernel_read_readvariableop:savev2_conv_lst_m2d_9_recurrent_kernel_read_readvariableop.savev2_conv_lst_m2d_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop7savev2_adam_conv_lst_m2d_9_kernel_m_read_readvariableopAsavev2_adam_conv_lst_m2d_9_recurrent_kernel_m_read_readvariableop5savev2_adam_conv_lst_m2d_9_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop7savev2_adam_conv_lst_m2d_9_kernel_v_read_readvariableopAsavev2_adam_conv_lst_m2d_9_recurrent_kernel_v_read_readvariableop5savev2_adam_conv_lst_m2d_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?1d:d:d:: : : : : :	?:@?:?: : : : :	?1d:d:d::	?:@?:?:	?1d:d:d::	?:@?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?1d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :-
)
'
_output_shapes
:	?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?1d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::-)
'
_output_shapes
:	?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?1d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::-)
'
_output_shapes
:	?:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:

_output_shapes
: 
?
?
.__inference_conv_lst_m2d_9_layer_call_fn_62543
inputs_0"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_607902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&??????????????????d	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&??????????????????d	
"
_user_specified_name
inputs/0
?

?
C__inference_dense_18_layer_call_and_return_conditional_losses_61522

inputs1
matmul_readvariableop_resource:	?1d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?1d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????1
 
_user_specified_nameinputs
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_61982
conv_lst_m2d_9_input/
conv_lst_m2d_9_61962:	?/
conv_lst_m2d_9_61964:@?#
conv_lst_m2d_9_61966:	?!
dense_18_61971:	?1d
dense_18_61973:d 
dense_19_61976:d
dense_19_61978:
identity??&conv_lst_m2d_9/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?
&conv_lst_m2d_9/StatefulPartitionedCallStatefulPartitionedCallconv_lst_m2d_9_inputconv_lst_m2d_9_61962conv_lst_m2d_9_61964conv_lst_m2d_9_61966*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_618472(
&conv_lst_m2d_9/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall/conv_lst_m2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_616092#
!dropout_9/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_615092
flatten_9/PartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_18_61971dense_18_61973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_615222"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_61976dense_19_61978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_615392"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0'^conv_lst_m2d_9/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 2P
&conv_lst_m2d_9/StatefulPartitionedCall&conv_lst_m2d_9/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:i e
3
_output_shapes!
:?????????d	
.
_user_specified_nameconv_lst_m2d_9_input
?o
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_61488

inputs8
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_8w
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_61362*
condR
while_cond_61361*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????d	: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
?
.__inference_conv_lst_m2d_9_layer_call_fn_62554
inputs_0"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_610282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&??????????????????d	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&??????????????????d	
"
_user_specified_name
inputs/0
?o
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63242

inputs8
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_8w
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_63116*
condR
while_cond_63115*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????d	: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
?
3__inference_conv_lst_m2d_cell_9_layer_call_fn_63576

inputs
states_0
states_1"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????b@:?????????b@:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_608962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????d	:?????????b@:?????????b@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????d	
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/1
?o
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63464

inputs8
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_8w
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_63338*
condR
while_cond_63337*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????d	: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
?
while_cond_61361
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_61361___redundant_placeholder03
/while_while_cond_61361___redundant_placeholder13
/while_while_cond_61361___redundant_placeholder23
/while_while_cond_61361___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_63479

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????b@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????b@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
?
?
(__inference_dense_19_layer_call_fn_63531

inputs
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_615392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?E
?
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_63651

inputs
states_0
states_18
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7w
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
IdentityIdentity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity?

Identity_1Identity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_1?

Identity_2Identity	add_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????d	:?????????b@:?????????b@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:?????????d	
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/1
?o
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_61847

inputs8
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_8w
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_61721*
condR
while_cond_61720*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????d	: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
?
3__inference_conv_lst_m2d_cell_9_layer_call_fn_63559

inputs
states_0
states_1"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????b@:?????????b@:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_607082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????d	:?????????b@:?????????b@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????d	
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????b@
"
_user_specified_name
states/1
?
?
while_cond_61720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_61720___redundant_placeholder03
/while_while_cond_61720___redundant_placeholder13
/while_while_cond_61720___redundant_placeholder23
/while_while_cond_61720___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?
E
)__inference_flatten_9_layer_call_fn_63496

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_615092
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs
?

?
C__inference_dense_19_layer_call_and_return_conditional_losses_63542

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?D
?
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_60708

inputs

states
states_18
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity

identity_1

identity_2??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOpd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution
BiasAddBiasAddconvolution:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
	BiasAdd_1BiasAddconvolution_1:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_2BiasAddconvolution_2:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_3BiasAddconvolution_3:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7w
addAddV2BiasAdd:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
IdentityIdentity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity?

Identity_1Identity	mul_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_1?

Identity_2Identity	add_5:z:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????d	:?????????b@:?????????b@: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp:W S
/
_output_shapes
:?????????d	
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????b@
 
_user_specified_namestates:WS
/
_output_shapes
:?????????b@
 
_user_specified_namestates
?
?
,sequential_9_conv_lst_m2d_9_while_cond_60460T
Psequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_loop_counterZ
Vsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_maximum_iterations1
-sequential_9_conv_lst_m2d_9_while_placeholder3
/sequential_9_conv_lst_m2d_9_while_placeholder_13
/sequential_9_conv_lst_m2d_9_while_placeholder_23
/sequential_9_conv_lst_m2d_9_while_placeholder_3T
Psequential_9_conv_lst_m2d_9_while_less_sequential_9_conv_lst_m2d_9_strided_slicek
gsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_cond_60460___redundant_placeholder0k
gsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_cond_60460___redundant_placeholder1k
gsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_cond_60460___redundant_placeholder2k
gsequential_9_conv_lst_m2d_9_while_sequential_9_conv_lst_m2d_9_while_cond_60460___redundant_placeholder3.
*sequential_9_conv_lst_m2d_9_while_identity
?
&sequential_9/conv_lst_m2d_9/while/LessLess-sequential_9_conv_lst_m2d_9_while_placeholderPsequential_9_conv_lst_m2d_9_while_less_sequential_9_conv_lst_m2d_9_strided_slice*
T0*
_output_shapes
: 2(
&sequential_9/conv_lst_m2d_9/while/Less?
*sequential_9/conv_lst_m2d_9/while/IdentityIdentity*sequential_9/conv_lst_m2d_9/while/Less:z:0*
T0
*
_output_shapes
: 2,
*sequential_9/conv_lst_m2d_9/while/Identity"a
*sequential_9_conv_lst_m2d_9_while_identity3sequential_9/conv_lst_m2d_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P: : : : :?????????b@:?????????b@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
:
?
?
.__inference_conv_lst_m2d_9_layer_call_fn_62576

inputs"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_618472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????d	: : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?
?
G__inference_sequential_9_layer_call_and_return_conditional_losses_61900

inputs/
conv_lst_m2d_9_61880:	?/
conv_lst_m2d_9_61882:@?#
conv_lst_m2d_9_61884:	?!
dense_18_61889:	?1d
dense_18_61891:d 
dense_19_61894:d
dense_19_61896:
identity??&conv_lst_m2d_9/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?
&conv_lst_m2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv_lst_m2d_9_61880conv_lst_m2d_9_61882conv_lst_m2d_9_61884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_618472(
&conv_lst_m2d_9/StatefulPartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall/conv_lst_m2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????b@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_616092#
!dropout_9/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_9_layer_call_and_return_conditional_losses_615092
flatten_9/PartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall"flatten_9/PartitionedCall:output:0dense_18_61889dense_18_61891*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_615222"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_61894dense_19_61896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_19_layer_call_and_return_conditional_losses_615392"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0'^conv_lst_m2d_9/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 2P
&conv_lst_m2d_9/StatefulPartitionedCall&conv_lst_m2d_9/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????d	
 
_user_specified_nameinputs
?h
?
while_body_61362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0@
%while_split_readvariableop_resource_0:	?B
'while_split_1_readvariableop_resource_0:@?6
'while_split_2_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor>
#while_split_readvariableop_resource:	?@
%while_split_1_readvariableop_resource:@?4
%while_split_2_readvariableop_resource:	???while/split/ReadVariableOp?while/split_1/ReadVariableOp?while/split_2/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:	?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
while/splitt
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
:@?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2
while/split_1t
while/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
while/split_2/split_dim?
while/split_2/ReadVariableOpReadVariableOp'while_split_2_readvariableop_resource_0*
_output_shapes	
:?*
dtype02
while/split_2/ReadVariableOp?
while/split_2Split while/split_2/split_dim:output:0$while/split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/split_2?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution?
while/BiasAddBiasAddwhile/convolution:output:0while/split_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_1?
while/BiasAdd_1BiasAddwhile/convolution_1:output:0while/split_2:output:1*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_2?
while/BiasAdd_2BiasAddwhile/convolution_2:output:0while/split_2:output:2*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
while/convolution_3?
while/BiasAdd_3BiasAddwhile/convolution_3:output:0while/split_2:output:3*
T0*/
_output_shapes
:?????????b@2
while/BiasAdd_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/BiasAdd:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????b@2
	while/add_
while/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Constc
while/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_1|
	while/MulMulwhile/add:z:0while/Const:output:0*
T0*/
_output_shapes
:?????????b@2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value?
while/add_2AddV2while/BiasAdd_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_2c
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3?
while/Mul_1Mulwhile/add_2:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????b@2
while/mul_2?
while/add_4AddV2while/BiasAdd_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_4k

while/ReluReluwhile/add_4:z:0*
T0*/
_output_shapes
:?????????b@2

while/Relu?
while/mul_3Mulwhile/clip_by_value:z:0while/Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
while/add_5?
while/add_6AddV2while/BiasAdd_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
while/add_6c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_4Mulwhile/add_6:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
while/clip_by_value_2o
while/Relu_1Reluwhile/add_5:z:0*
T0*/
_output_shapes
:?????????b@2
while/Relu_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9?
while/IdentityIdentitywhile/add_9:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add_8:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/mul_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identitywhile/add_5:z:0^while/split/ReadVariableOp^while/split_1/ReadVariableOp^while/split_2/ReadVariableOp*
T0*/
_output_shapes
:?????????b@2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"P
%while_split_2_readvariableop_resource'while_split_2_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 28
while/split/ReadVariableOpwhile/split/ReadVariableOp2<
while/split_1/ReadVariableOpwhile/split_1/ReadVariableOp2<
while/split_2/ReadVariableOpwhile/split_2/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_dense_18_layer_call_fn_63511

inputs
unknown:	?1d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_615222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????1
 
_user_specified_nameinputs
?#
?
while_body_60960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_60984_0:	?(
while_60986_0:@?
while_60988_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_60984:	?&
while_60986:@?
while_60988:	???while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????d	*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_60984_0while_60986_0while_60988_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????b@:?????????b@:?????????b@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_608962
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1~
while/IdentityIdentitywhile/add_1:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????b@2
while/Identity_5"
while_60984while_60984_0"
while_60986while_60986_0"
while_60988while_60988_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : :?????????b@:?????????b@: : : : : 2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????b@:51
/
_output_shapes
:?????????b@:

_output_shapes
: :

_output_shapes
: 
?	
?
,__inference_sequential_9_layer_call_fn_61936
conv_lst_m2d_9_input"
unknown:	?$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?1d
	unknown_3:d
	unknown_4:d
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv_lst_m2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_9_layer_call_and_return_conditional_losses_619002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????d	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
3
_output_shapes!
:?????????d	
.
_user_specified_nameconv_lst_m2d_9_input
?p
?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63020
inputs_08
split_readvariableop_resource:	?:
split_1_readvariableop_resource:@?.
split_2_readvariableop_resource:	?
identity??split/ReadVariableOp?split_1/ReadVariableOp?split_2/ReadVariableOp?whilev

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&??????????????????d	2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????d	2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"      	   @   2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
:	@2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????d	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   d   	   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????d	*
shrink_axis_mask2
strided_slice_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:	?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:	@:	@:	@:	@*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
:@?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:@@:@@:@@:@@*
	num_split2	
split_1h
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dim?
split_2/ReadVariableOpReadVariableOpsplit_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_2/ReadVariableOp?
split_2Splitsplit_2/split_dim:output:0split_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_2?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_1?
BiasAddBiasAddconvolution_1:output:0split_2:output:0*
T0*/
_output_shapes
:?????????b@2	
BiasAdd?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_2?
	BiasAdd_1BiasAddconvolution_2:output:0split_2:output:1*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_1?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_3?
	BiasAdd_2BiasAddconvolution_3:output:0split_2:output:2*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_2?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????b@*
paddingVALID*
strides
2
convolution_4?
	BiasAdd_3BiasAddconvolution_4:output:0split_2:output:3*
T0*/
_output_shapes
:?????????b@2
	BiasAdd_3?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????b@*
paddingSAME*
strides
2
convolution_8w
addAddV2BiasAdd:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????b@2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1d
MulMuladd:z:0Const:output:0*
T0*/
_output_shapes
:?????????b@2
Mulj
Add_1AddMul:z:0Const_1:output:0*
T0*/
_output_shapes
:?????????b@2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value}
add_2AddV2BiasAdd_1:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????b@2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3l
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_1l
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????b@2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????b@2
mul_2}
add_4AddV2BiasAdd_2:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????b@2
add_4Y
ReluRelu	add_4:z:0*
T0*/
_output_shapes
:?????????b@2
Reluv
mul_3Mulclip_by_value:z:0Relu:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????b@2
add_5}
add_6AddV2BiasAdd_3:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????b@2
add_6W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????b@2
Mul_4l
Add_7Add	Mul_4:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????b@2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????b@2
clip_by_value_2]
Relu_1Relu	add_5:z:0*
T0*/
_output_shapes
:?????????b@2
Relu_1z
mul_5Mulclip_by_value_2:z:0Relu_1:activations:0*
T0*/
_output_shapes
:?????????b@2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcesplit_2_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*\
_output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_62894*
condR
while_cond_62893*[
output_shapesJ
H: : : : :?????????b@:?????????b@: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????   b   @   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????b@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????b@*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????b@2
transpose_1?
IdentityIdentitystrided_slice_2:output:0^split/ReadVariableOp^split_1/ReadVariableOp^split_2/ReadVariableOp^while*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:&??????????????????d	: : : 2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
split_2/ReadVariableOpsplit_2/ReadVariableOp2
whilewhile:f b
<
_output_shapes*
(:&??????????????????d	
"
_user_specified_name
inputs/0
?
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_61609

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????b@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????b@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????b@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????b@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????b@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????b@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????b@:W S
/
_output_shapes
:?????????b@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
a
conv_lst_m2d_9_inputI
&serving_default_conv_lst_m2d_9_input:0?????????d	<
dense_190
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?7
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
o__call__
*p&call_and_return_all_conditional_losses
q_default_save_signature"?4
_tf_keras_sequential?4{"name": "sequential_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 1, 100, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv_lst_m2d_9_input"}}, {"class_name": "ConvLSTM2D", "config": {"name": "conv_lst_m2d_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 1, 100, 9]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1, 100, 9]}, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 15}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 1, 100, 9]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 4, 1, 100, 9]}, "float32", "conv_lst_m2d_9_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 1, 100, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv_lst_m2d_9_input"}, "shared_object_id": 0}, {"class_name": "ConvLSTM2D", "config": {"name": "conv_lst_m2d_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 1, 100, 9]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "shared_object_id": 5}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 6}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 16}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?{"name": "conv_lst_m2d_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 1, 100, 9]}, "stateful": false, "must_restore_from_config": false, "class_name": "ConvLSTM2D", "config": {"name": "conv_lst_m2d_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 1, 100, 9]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "shared_object_id": 5, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1, 100, 9]}, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 15}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 1, 100, 9]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 6}
?
	variables
regularization_losses
trainable_variables
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 17}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6272}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6272]}}
?

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
&iter

'beta_1

(beta_2
	)decay
*learning_ratemamb mc!md+me,mf-mgvhvi vj!vk+vl,vm-vn"
	optimizer
Q
+0
,1
-2
3
4
 5
!6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
+0
,1
-2
3
4
 5
!6"
trackable_list_wrapper
?

.layers
/layer_metrics
0metrics
	variables
1layer_regularization_losses
regularization_losses
2non_trainable_variables
	trainable_variables
o__call__
q_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
,
|serving_default"
signature_map
?


+kernel
,recurrent_kernel
-bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv_lst_m2d_cell_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvLSTM2DCell", "config": {"name": "conv_lst_m2d_cell_9", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "shared_object_id": 4}
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?

7layers
8layer_metrics
9metrics
	variables

:states
;layer_regularization_losses
regularization_losses
<non_trainable_variables
trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

=layers
>layer_metrics
?metrics
	variables
@layer_regularization_losses
regularization_losses
Anon_trainable_variables
trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Blayers
Clayer_metrics
Dmetrics
	variables
Elayer_regularization_losses
regularization_losses
Fnon_trainable_variables
trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
": 	?1d2dense_18/kernel
:d2dense_18/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Glayers
Hlayer_metrics
Imetrics
	variables
Jlayer_regularization_losses
regularization_losses
Knon_trainable_variables
trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_19/kernel
:2dense_19/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?

Llayers
Mlayer_metrics
Nmetrics
"	variables
Olayer_regularization_losses
#regularization_losses
Pnon_trainable_variables
$trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0:.	?2conv_lst_m2d_9/kernel
::8@?2conv_lst_m2d_9/recurrent_kernel
": ?2conv_lst_m2d_9/bias
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?

Slayers
Tlayer_metrics
Umetrics
3	variables
Vlayer_regularization_losses
4regularization_losses
Wnon_trainable_variables
5trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Xtotal
	Ycount
Z	variables
[	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 20}
?
	\total
	]count
^
_fn_kwargs
_	variables
`	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 16}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
X0
Y1"
trackable_list_wrapper
-
Z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
':%	?1d2Adam/dense_18/kernel/m
 :d2Adam/dense_18/bias/m
&:$d2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
5:3	?2Adam/conv_lst_m2d_9/kernel/m
?:=@?2&Adam/conv_lst_m2d_9/recurrent_kernel/m
':%?2Adam/conv_lst_m2d_9/bias/m
':%	?1d2Adam/dense_18/kernel/v
 :d2Adam/dense_18/bias/v
&:$d2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
5:3	?2Adam/conv_lst_m2d_9/kernel/v
?:=@?2&Adam/conv_lst_m2d_9/recurrent_kernel/v
':%?2Adam/conv_lst_m2d_9/bias/v
?2?
,__inference_sequential_9_layer_call_fn_61563
,__inference_sequential_9_layer_call_fn_62028
,__inference_sequential_9_layer_call_fn_62047
,__inference_sequential_9_layer_call_fn_61936?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_9_layer_call_and_return_conditional_losses_62286
G__inference_sequential_9_layer_call_and_return_conditional_losses_62532
G__inference_sequential_9_layer_call_and_return_conditional_losses_61959
G__inference_sequential_9_layer_call_and_return_conditional_losses_61982?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_60604?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *??<
:?7
conv_lst_m2d_9_input?????????d	
?2?
.__inference_conv_lst_m2d_9_layer_call_fn_62543
.__inference_conv_lst_m2d_9_layer_call_fn_62554
.__inference_conv_lst_m2d_9_layer_call_fn_62565
.__inference_conv_lst_m2d_9_layer_call_fn_62576?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_62798
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63020
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63242
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63464?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_9_layer_call_fn_63469
)__inference_dropout_9_layer_call_fn_63474?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_9_layer_call_and_return_conditional_losses_63479
D__inference_dropout_9_layer_call_and_return_conditional_losses_63491?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_flatten_9_layer_call_fn_63496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_9_layer_call_and_return_conditional_losses_63502?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_18_layer_call_fn_63511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_18_layer_call_and_return_conditional_losses_63522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_19_layer_call_fn_63531?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_19_layer_call_and_return_conditional_losses_63542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_62009conv_lst_m2d_9_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_conv_lst_m2d_cell_9_layer_call_fn_63559
3__inference_conv_lst_m2d_cell_9_layer_call_fn_63576?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_63651
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_63726?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_60604?+,- !I?F
??<
:?7
conv_lst_m2d_9_input?????????d	
? "3?0
.
dense_19"?
dense_19??????????
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_62798?+,-W?T
M?J
<?9
7?4
inputs/0&??????????????????d	

 
p 

 
? "-?*
#? 
0?????????b@
? ?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63020?+,-W?T
M?J
<?9
7?4
inputs/0&??????????????????d	

 
p

 
? "-?*
#? 
0?????????b@
? ?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63242}+,-G?D
=?:
,?)
inputs?????????d	

 
p 

 
? "-?*
#? 
0?????????b@
? ?
I__inference_conv_lst_m2d_9_layer_call_and_return_conditional_losses_63464}+,-G?D
=?:
,?)
inputs?????????d	

 
p

 
? "-?*
#? 
0?????????b@
? ?
.__inference_conv_lst_m2d_9_layer_call_fn_62543?+,-W?T
M?J
<?9
7?4
inputs/0&??????????????????d	

 
p 

 
? " ??????????b@?
.__inference_conv_lst_m2d_9_layer_call_fn_62554?+,-W?T
M?J
<?9
7?4
inputs/0&??????????????????d	

 
p

 
? " ??????????b@?
.__inference_conv_lst_m2d_9_layer_call_fn_62565p+,-G?D
=?:
,?)
inputs?????????d	

 
p 

 
? " ??????????b@?
.__inference_conv_lst_m2d_9_layer_call_fn_62576p+,-G?D
=?:
,?)
inputs?????????d	

 
p

 
? " ??????????b@?
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_63651?+,-???
???
(?%
inputs?????????d	
[?X
*?'
states/0?????????b@
*?'
states/1?????????b@
p 
? "???
??~
%?"
0/0?????????b@
U?R
'?$
0/1/0?????????b@
'?$
0/1/1?????????b@
? ?
N__inference_conv_lst_m2d_cell_9_layer_call_and_return_conditional_losses_63726?+,-???
???
(?%
inputs?????????d	
[?X
*?'
states/0?????????b@
*?'
states/1?????????b@
p
? "???
??~
%?"
0/0?????????b@
U?R
'?$
0/1/0?????????b@
'?$
0/1/1?????????b@
? ?
3__inference_conv_lst_m2d_cell_9_layer_call_fn_63559?+,-???
???
(?%
inputs?????????d	
[?X
*?'
states/0?????????b@
*?'
states/1?????????b@
p 
? "{?x
#? 
0?????????b@
Q?N
%?"
1/0?????????b@
%?"
1/1?????????b@?
3__inference_conv_lst_m2d_cell_9_layer_call_fn_63576?+,-???
???
(?%
inputs?????????d	
[?X
*?'
states/0?????????b@
*?'
states/1?????????b@
p
? "{?x
#? 
0?????????b@
Q?N
%?"
1/0?????????b@
%?"
1/1?????????b@?
C__inference_dense_18_layer_call_and_return_conditional_losses_63522]0?-
&?#
!?
inputs??????????1
? "%?"
?
0?????????d
? |
(__inference_dense_18_layer_call_fn_63511P0?-
&?#
!?
inputs??????????1
? "??????????d?
C__inference_dense_19_layer_call_and_return_conditional_losses_63542\ !/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? {
(__inference_dense_19_layer_call_fn_63531O !/?,
%?"
 ?
inputs?????????d
? "???????????
D__inference_dropout_9_layer_call_and_return_conditional_losses_63479l;?8
1?.
(?%
inputs?????????b@
p 
? "-?*
#? 
0?????????b@
? ?
D__inference_dropout_9_layer_call_and_return_conditional_losses_63491l;?8
1?.
(?%
inputs?????????b@
p
? "-?*
#? 
0?????????b@
? ?
)__inference_dropout_9_layer_call_fn_63469_;?8
1?.
(?%
inputs?????????b@
p 
? " ??????????b@?
)__inference_dropout_9_layer_call_fn_63474_;?8
1?.
(?%
inputs?????????b@
p
? " ??????????b@?
D__inference_flatten_9_layer_call_and_return_conditional_losses_63502a7?4
-?*
(?%
inputs?????????b@
? "&?#
?
0??????????1
? ?
)__inference_flatten_9_layer_call_fn_63496T7?4
-?*
(?%
inputs?????????b@
? "???????????1?
G__inference_sequential_9_layer_call_and_return_conditional_losses_61959?+,- !Q?N
G?D
:?7
conv_lst_m2d_9_input?????????d	
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_61982?+,- !Q?N
G?D
:?7
conv_lst_m2d_9_input?????????d	
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_62286u+,- !C?@
9?6
,?)
inputs?????????d	
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_9_layer_call_and_return_conditional_losses_62532u+,- !C?@
9?6
,?)
inputs?????????d	
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_9_layer_call_fn_61563v+,- !Q?N
G?D
:?7
conv_lst_m2d_9_input?????????d	
p 

 
? "???????????
,__inference_sequential_9_layer_call_fn_61936v+,- !Q?N
G?D
:?7
conv_lst_m2d_9_input?????????d	
p

 
? "???????????
,__inference_sequential_9_layer_call_fn_62028h+,- !C?@
9?6
,?)
inputs?????????d	
p 

 
? "???????????
,__inference_sequential_9_layer_call_fn_62047h+,- !C?@
9?6
,?)
inputs?????????d	
p

 
? "???????????
#__inference_signature_wrapper_62009?+,- !a?^
? 
W?T
R
conv_lst_m2d_9_input:?7
conv_lst_m2d_9_input?????????d	"3?0
.
dense_19"?
dense_19?????????