กพ(
แ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

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
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
ฎ
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ม
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
executor_typestring จ
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
๗
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018ค่%
จ
(Adam/transformer_model_17/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/transformer_model_17/dense_1/bias/v
ก
<Adam/transformer_model_17/dense_1/bias/v/Read/ReadVariableOpReadVariableOp(Adam/transformer_model_17/dense_1/bias/v*
_output_shapes
:*
dtype0
ฑ
*Adam/transformer_model_17/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/transformer_model_17/dense_1/kernel/v
ช
>Adam/transformer_model_17/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_model_17/dense_1/kernel/v*
_output_shapes
:	*
dtype0
ฅ
&Adam/transformer_model_17/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/transformer_model_17/dense/bias/v

:Adam/transformer_model_17/dense/bias/v/Read/ReadVariableOpReadVariableOp&Adam/transformer_model_17/dense/bias/v*
_output_shapes	
:*
dtype0
ฎ
(Adam/transformer_model_17/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(Adam/transformer_model_17/dense/kernel/v
ง
<Adam/transformer_model_17/dense/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/transformer_model_17/dense/kernel/v* 
_output_shapes
:
*
dtype0
ํ
JAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/v
ๆ
^Adam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOpJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/v*
_output_shapes	
:*
dtype0
๏
KAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*\
shared_nameMKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/v
่
_Adam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOpKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/v*
_output_shapes	
:*
dtype0
ำ
=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/v
ฬ
QAdam/transformer_model_17/transformer_encoder/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/v*
_output_shapes	
:*
dtype0
เ
?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/v
ู
SAdam/transformer_model_17/transformer_encoder/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/v*$
_output_shapes
:*
dtype0
ฯ
;Adam/transformer_model_17/transformer_encoder/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;Adam/transformer_model_17/transformer_encoder/conv1d/bias/v
ศ
OAdam/transformer_model_17/transformer_encoder/conv1d/bias/v/Read/ReadVariableOpReadVariableOp;Adam/transformer_model_17/transformer_encoder/conv1d/bias/v*
_output_shapes	
:*
dtype0
?
=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/v
ี
QAdam/transformer_model_17/transformer_encoder/conv1d/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/v*$
_output_shapes
:*
dtype0
้
HAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/v
โ
\Adam/transformer_model_17/transformer_encoder/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOpHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/v*
_output_shapes	
:*
dtype0
๋
IAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/v
ไ
]Adam/transformer_model_17/transformer_encoder/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/v*
_output_shapes	
:*
dtype0

TAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*e
shared_nameVTAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/v
๚
hAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/v/Read/ReadVariableOpReadVariableOpTAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/v*
_output_shapes	
:*
dtype0

VAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*g
shared_nameXVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/v

jAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/v/Read/ReadVariableOpReadVariableOpVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/v*$
_output_shapes
:*
dtype0

QAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/v
?
eAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/v/Read/ReadVariableOpReadVariableOpQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/v*$
_output_shapes
:*
dtype0

OAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*`
shared_nameQOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/v
๙
cAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/v/Read/ReadVariableOpReadVariableOpOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/v*$
_output_shapes
:*
dtype0

QAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/v
?
eAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/v/Read/ReadVariableOpReadVariableOpQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/v*$
_output_shapes
:*
dtype0
ำ
=Adam/transformer_model_17/positional_embedding/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_model_17/positional_embedding/dense_2/bias/v
ฬ
QAdam/transformer_model_17/positional_embedding/dense_2/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_model_17/positional_embedding/dense_2/bias/v*
_output_shapes	
:*
dtype0
?
?Adam/transformer_model_17/positional_embedding/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?Adam/transformer_model_17/positional_embedding/dense_2/kernel/v
ิ
SAdam/transformer_model_17/positional_embedding/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_model_17/positional_embedding/dense_2/kernel/v*
_output_shapes
:	*
dtype0
จ
(Adam/transformer_model_17/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/transformer_model_17/dense_1/bias/m
ก
<Adam/transformer_model_17/dense_1/bias/m/Read/ReadVariableOpReadVariableOp(Adam/transformer_model_17/dense_1/bias/m*
_output_shapes
:*
dtype0
ฑ
*Adam/transformer_model_17/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/transformer_model_17/dense_1/kernel/m
ช
>Adam/transformer_model_17/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_model_17/dense_1/kernel/m*
_output_shapes
:	*
dtype0
ฅ
&Adam/transformer_model_17/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/transformer_model_17/dense/bias/m

:Adam/transformer_model_17/dense/bias/m/Read/ReadVariableOpReadVariableOp&Adam/transformer_model_17/dense/bias/m*
_output_shapes	
:*
dtype0
ฎ
(Adam/transformer_model_17/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(Adam/transformer_model_17/dense/kernel/m
ง
<Adam/transformer_model_17/dense/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/transformer_model_17/dense/kernel/m* 
_output_shapes
:
*
dtype0
ํ
JAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/m
ๆ
^Adam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOpJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/m*
_output_shapes	
:*
dtype0
๏
KAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*\
shared_nameMKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/m
่
_Adam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOpKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/m*
_output_shapes	
:*
dtype0
ำ
=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/m
ฬ
QAdam/transformer_model_17/transformer_encoder/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/m*
_output_shapes	
:*
dtype0
เ
?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/m
ู
SAdam/transformer_model_17/transformer_encoder/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/m*$
_output_shapes
:*
dtype0
ฯ
;Adam/transformer_model_17/transformer_encoder/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;Adam/transformer_model_17/transformer_encoder/conv1d/bias/m
ศ
OAdam/transformer_model_17/transformer_encoder/conv1d/bias/m/Read/ReadVariableOpReadVariableOp;Adam/transformer_model_17/transformer_encoder/conv1d/bias/m*
_output_shapes	
:*
dtype0
?
=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/m
ี
QAdam/transformer_model_17/transformer_encoder/conv1d/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/m*$
_output_shapes
:*
dtype0
้
HAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/m
โ
\Adam/transformer_model_17/transformer_encoder/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOpHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/m*
_output_shapes	
:*
dtype0
๋
IAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/m
ไ
]Adam/transformer_model_17/transformer_encoder/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/m*
_output_shapes	
:*
dtype0

TAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*e
shared_nameVTAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/m
๚
hAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/m/Read/ReadVariableOpReadVariableOpTAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/m*
_output_shapes	
:*
dtype0

VAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*g
shared_nameXVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/m

jAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/m/Read/ReadVariableOpReadVariableOpVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/m*$
_output_shapes
:*
dtype0

QAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/m
?
eAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/m/Read/ReadVariableOpReadVariableOpQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/m*$
_output_shapes
:*
dtype0

OAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*`
shared_nameQOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/m
๙
cAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/m/Read/ReadVariableOpReadVariableOpOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/m*$
_output_shapes
:*
dtype0

QAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*b
shared_nameSQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/m
?
eAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/m/Read/ReadVariableOpReadVariableOpQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/m*$
_output_shapes
:*
dtype0
ำ
=Adam/transformer_model_17/positional_embedding/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_model_17/positional_embedding/dense_2/bias/m
ฬ
QAdam/transformer_model_17/positional_embedding/dense_2/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_model_17/positional_embedding/dense_2/bias/m*
_output_shapes	
:*
dtype0
?
?Adam/transformer_model_17/positional_embedding/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*P
shared_nameA?Adam/transformer_model_17/positional_embedding/dense_2/kernel/m
ิ
SAdam/transformer_model_17/positional_embedding/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_model_17/positional_embedding/dense_2/kernel/m*
_output_shapes
:	*
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
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
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0

!transformer_model_17/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!transformer_model_17/dense_1/bias

5transformer_model_17/dense_1/bias/Read/ReadVariableOpReadVariableOp!transformer_model_17/dense_1/bias*
_output_shapes
:*
dtype0
ฃ
#transformer_model_17/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#transformer_model_17/dense_1/kernel

7transformer_model_17/dense_1/kernel/Read/ReadVariableOpReadVariableOp#transformer_model_17/dense_1/kernel*
_output_shapes
:	*
dtype0

transformer_model_17/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!transformer_model_17/dense/bias

3transformer_model_17/dense/bias/Read/ReadVariableOpReadVariableOptransformer_model_17/dense/bias*
_output_shapes	
:*
dtype0
?
!transformer_model_17/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!transformer_model_17/dense/kernel

5transformer_model_17/dense/kernel/Read/ReadVariableOpReadVariableOp!transformer_model_17/dense/kernel* 
_output_shapes
:
*
dtype0
฿
Ctransformer_model_17/transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECtransformer_model_17/transformer_encoder/layer_normalization_1/beta
ุ
Wtransformer_model_17/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOpCtransformer_model_17/transformer_encoder/layer_normalization_1/beta*
_output_shapes	
:*
dtype0
แ
Dtransformer_model_17/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDtransformer_model_17/transformer_encoder/layer_normalization_1/gamma
ฺ
Xtransformer_model_17/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOpDtransformer_model_17/transformer_encoder/layer_normalization_1/gamma*
_output_shapes	
:*
dtype0
ล
6transformer_model_17/transformer_encoder/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86transformer_model_17/transformer_encoder/conv1d_1/bias
พ
Jtransformer_model_17/transformer_encoder/conv1d_1/bias/Read/ReadVariableOpReadVariableOp6transformer_model_17/transformer_encoder/conv1d_1/bias*
_output_shapes	
:*
dtype0
า
8transformer_model_17/transformer_encoder/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_model_17/transformer_encoder/conv1d_1/kernel
ห
Ltransformer_model_17/transformer_encoder/conv1d_1/kernel/Read/ReadVariableOpReadVariableOp8transformer_model_17/transformer_encoder/conv1d_1/kernel*$
_output_shapes
:*
dtype0
ม
4transformer_model_17/transformer_encoder/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64transformer_model_17/transformer_encoder/conv1d/bias
บ
Htransformer_model_17/transformer_encoder/conv1d/bias/Read/ReadVariableOpReadVariableOp4transformer_model_17/transformer_encoder/conv1d/bias*
_output_shapes	
:*
dtype0
ฮ
6transformer_model_17/transformer_encoder/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86transformer_model_17/transformer_encoder/conv1d/kernel
ว
Jtransformer_model_17/transformer_encoder/conv1d/kernel/Read/ReadVariableOpReadVariableOp6transformer_model_17/transformer_encoder/conv1d/kernel*$
_output_shapes
:*
dtype0
?
Atransformer_model_17/transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAtransformer_model_17/transformer_encoder/layer_normalization/beta
ิ
Utransformer_model_17/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOpAtransformer_model_17/transformer_encoder/layer_normalization/beta*
_output_shapes	
:*
dtype0
?
Btransformer_model_17/transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBtransformer_model_17/transformer_encoder/layer_normalization/gamma
ึ
Vtransformer_model_17/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOpBtransformer_model_17/transformer_encoder/layer_normalization/gamma*
_output_shapes	
:*
dtype0
๓
Mtransformer_model_17/transformer_encoder/multi_head_attention/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*^
shared_nameOMtransformer_model_17/transformer_encoder/multi_head_attention/projection_bias
์
atransformer_model_17/transformer_encoder/multi_head_attention/projection_bias/Read/ReadVariableOpReadVariableOpMtransformer_model_17/transformer_encoder/multi_head_attention/projection_bias*
_output_shapes	
:*
dtype0

Otransformer_model_17/transformer_encoder/multi_head_attention/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*`
shared_nameQOtransformer_model_17/transformer_encoder/multi_head_attention/projection_kernel
๙
ctransformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/Read/ReadVariableOpReadVariableOpOtransformer_model_17/transformer_encoder/multi_head_attention/projection_kernel*$
_output_shapes
:*
dtype0
๖
Jtransformer_model_17/transformer_encoder/multi_head_attention/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJtransformer_model_17/transformer_encoder/multi_head_attention/value_kernel
๏
^transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/Read/ReadVariableOpReadVariableOpJtransformer_model_17/transformer_encoder/multi_head_attention/value_kernel*$
_output_shapes
:*
dtype0
๒
Htransformer_model_17/transformer_encoder/multi_head_attention/key_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHtransformer_model_17/transformer_encoder/multi_head_attention/key_kernel
๋
\transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/Read/ReadVariableOpReadVariableOpHtransformer_model_17/transformer_encoder/multi_head_attention/key_kernel*$
_output_shapes
:*
dtype0
๖
Jtransformer_model_17/transformer_encoder/multi_head_attention/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJtransformer_model_17/transformer_encoder/multi_head_attention/query_kernel
๏
^transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/Read/ReadVariableOpReadVariableOpJtransformer_model_17/transformer_encoder/multi_head_attention/query_kernel*$
_output_shapes
:*
dtype0
ล
6transformer_model_17/positional_embedding/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86transformer_model_17/positional_embedding/dense_2/bias
พ
Jtransformer_model_17/positional_embedding/dense_2/bias/Read/ReadVariableOpReadVariableOp6transformer_model_17/positional_embedding/dense_2/bias*
_output_shapes	
:*
dtype0
อ
8transformer_model_17/positional_embedding/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*I
shared_name:8transformer_model_17/positional_embedding/dense_2/kernel
ฦ
Ltransformer_model_17/positional_embedding/dense_2/kernel/Read/ReadVariableOpReadVariableOp8transformer_model_17/positional_embedding/dense_2/kernel*
_output_shapes
:	*
dtype0
โ
ConstConst* 
_output_shapes
:
*
dtype0*ก
valueB
"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?คjW?^MM? C?ฎ8?ฮz.?~$?:อ?ut?บ}?๛??>ไ๏>G(เ>๐ั>)อร>`ืถ>ฎจช>>9>ฃ>v>ต>'p>ู#`>aฮP>(~B>:#5>Rฎ(>ไ>!=>๓%>~?=]๙๋=จ?=vuฬ=dNพ=ฟ ฑ=ก?ค=7o=ผฬ=`ๆ=^w=ฐ6f=ู>V=bG=ึ9=๛ฌ,=ฑ =ฦ=๐)=\=๑<ฅMเ<฿ปะ<ฆ>ย<ะยด<6จ<&<Bซ<r<ฆJ|<ูฦj<IzZ<^OK<?1=<t0<Wึ#<uv<นเ<<น๕;ฬฉไ;ฒษิ;ฟฦ;\Dธ;Pyซ;ฃ;};].;h;๊Qo;^ด^; >O;ณฺ@; w3;G';i;ำ;i;็x๚: ้:lๆุ:Wืษ:่ำป:ษฎ:๓ฆข:\:ูู:m:?๑s:เc:&?S:D:๎6:C;*:คi:j:.	:ฌO?9โํ9%?9ฝอ9ืtฟ9๛)ฒ9_หฅ9H9r9ฉ9จx9dg9ไSW9ม`H9Sw:9(-9y!9#C9qิ9+9ใ,๒8\แ8@Q
?ํ?)?%?๐G1?ฐT;?,(D?ชไK?ถฉR?X?จฝ]?ี=b?`)f?ฦi?dl?ฉo?I\q?jNs?ิ?t?vv?ปw?๐ีx?ลสy?	z?	W{?๖{?ธ|?|๘|?@`}?(บ}?~?K~?๗~?ธ~?rไ~?n
?S+?ัG?`??u?`?h?Jฆ?Oฒ?นผ?ฝล?อ?Oิ?*ฺ?<฿??ใ?n็?น๊?ํ?๐?/๒?	๔?ฃ๕?๗?;๘?E๙?,๚?๔๚?ก๛?7??น??*????฿??(??h????ฮ??๗????9??T??k????????ฌ??ท??ม??ส??ั??ื?????แ??ๅ??้??์??๏??๑??๓??๕??๖??๘??๙??๚??๛??๛???????????????????????????????????????????????  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?ทวh?๓Hu?ฏ|?:ษ?ูZ?|?v?๊co?
๒f?1?]?ฝS?kI?5??๛๊4?๙ศ*?Qๆ ?T?m?VM?fห๙>ถำ้>gดฺ>Bkฬ>ะ๓พ>๖Gฒ>i`ฆ>5>eฝ>๐>{>gj>ฉcZ>๙oK>{=>ึv0>ฌR$>ำ >รs>ช>๋๊๖=ูๅ=ล๓ี=%ว=\น=ฎฌ=โ?=@q=_=ศp=ืํp=8`=ชP=ซ0B=ท4=o-(=ี=\ฅ=ฒ=?B?<ญภ๊<Puฺ<\Kห<ร.ฝ<ูฐ<>ิฃ<ฤt<\฿<<Tทu<_จd<ศT<าF<C8<ถx+<'<)}<.
<' <Q๏;	ด?;?=ฯ;|ฺภ;ิvณ;$ง;ๆh;ผ;V;ษxz;i;YๆX;GืI;?ำ;;ษ.;๋ฆ";\;ิู;i;๑๓:ฺใ:!?ำ:ฤ:๎ถ:@;ช:ขi:j:.:ชO:แm:$]:ฝM:ึt?:๚)2:_ห%:H:r:ฉ:จ๘9d็9ไSื9ม`ศ9Swบ9(ญ9yก9#C9qิ9+9ใ,r9\a93ีพพSI$พHn'ฝQM=Kl2>ฦ>|kต>ฒ่?>" ?uโ?พ??9*?็5?ด>?ลG?์{N?ํT?<Z?w_?iพc?(xg? ถj?m?ฅ๚o?Xr?I๔s?วu?๕๒v?ํ'x?โ3y?9z?ฅๅz?<{?+|?ฏฎ|?O }?ย}?ุ}?๑!~?๓a~?f~?mษ~?๓~??C6?JQ?ณh?๙|??ผ?็ช?Nถ?/ภ?ผศ?$ะ?ึ???์เ?ๅ?ฑ่?ั๋?๎??๐?ๅ๒?ฆ๔?,๖?}๗?ก๘?๙?y๚?7๛??๛?i??ไ??O??ซ??๛??A??}??ฐ???????&??C??]??s??????ค??ฐ??ป??ฤ??ฬ??ำ??ู?????ใ??็??๊??ํ??๐??๒??๔??๕??๗??๘??๙??๚??๛??????????????????????????????????????????????????  ?  ?ร>ฏ>&n?j)??<G?kั]?4?m??x?ืE~?๙??ฌt~?!Nz?c!t?Fnl?z?c?OZ?ย	P?ฤE?p;?๑11?ธ$'?ฝ]?$์?oฺ
?m/?๗?๓>\5ไ>?dี>งhว>Q<บ>ูญ>)7ข>mN> >_>&u>โmd>คฯT>;F>๊8>ุ๎+> >>'ศ
>,3>ฒ๐=๊฿=ฉlะ=ม?ม=?ด=/จ=6j=={=u-|=vฐj=MiZ=ชBK=ฌ(==ห0=ลั#=s=	฿=b= น๕<ํชไ<ฃหิ<Eฦ<IGธ<|ซ<๚<๖<ว1<ศ<Xo<รบ^<IDO<เ@<|3<'<n<ฃ<่<]๚;้;เํุ;S?ษ;qฺป;จฯฎ;ชฌข;ba;ี?;;L๚s;๗	c;ฐFS;D;*๕6;]A*;So;ao;๐2	;?X?:oํ:?:?ฤอ:ฝ{ฟ:g0ฒ:Yัฅ:.N:?:{:ฑx:๊lg:ช[W:?gH:~::l-:๋~!:H:~ู:?#:ก5๒9ตdแ9ชพั9ม.ร9ชกต9mฉ9&p}ฟ}pฟม[ฟ9ฦ?ฟอฟ ฟo?พaฆผพ}uพจํฝดฉk:เ=6ษV>c>ำTฤ>2H๊>R?W/?ำ"?โ\.?ณร8?ข็A?ฆ๋I?๐P?QW?)m\?a?	*e?ดh?7ษk?ศvn?vสp?อฯr?๏t?ถv?โhw?4x?y?iz?K({?ฮ{?]|?ฺ|?ๅE}?Sฃ}?C๔}?b:~?w~?ผซ~?Oู~?ศ ?๙"?@?;Z?qp?ญ?V?รข?Bฏ?บ?sร?ห?า?ญุ?๒??โ?wๆ?ใ้?ฺ์?k๏?ค๑?๓?;๕?ญ๖?ํ๗?๙?๒๙?ม๚?u๛???????r??ส????W????ย??์????1??M??e??z??????ฉ??ต??ฟ??ว??ฯ??ึ?????เ??ไ??่??๋??๎??๐??๓??๔??๖??๗??๘??๙??๚??๛??????????????????????????????????????ฯฝAฟjqฟ(ขพuJงฝ๏>ฑฏ>ณ?'ฆ)?ํIG??]?ึn?แx?ฅG~?๕??0s~?~Kz??t?jl?ืc?_Z?ฆP?pฟE?^k;?ื,1?ผ'?้X?~็?๛ี
?-+?ๅี๓>ถ-ไ>f]ี>ืaว>่5บ> ำญ>1ข> I>*>ป>าu>ฦed>ศT>๕3F>R8>ฒ่+>Y >C>,ร
>.>๐=โ฿=)eะ=ล๘ม=ด=!	จ=d=ศ=ฑv=]$|=?งj=laZ=U;K=ุ!==q0=?ห#=n=ู๊==Aฐ๕<ฌขไ<๔ริ<?ล<ข@ธ<Nvซ<7<{<ส,<#<เOo<ธฒ^<อ<O<กู@<$v3< '<th<`<<Qx๚;ง้;ๆุ;ืษ;ฉำป;Xษฎ;สฆข;๊[;ฟู;X;}๑s;ฤc;?S;yD;๎6;7;*;i;j;?-	;ขO?:ฺํ:?:ฝอ:ำtฟ:๗)ฒ:\หฅ:H:p:ง:จx:dg:ใSW:ภ`H:Rw::(-:y!:#C:qิ:+:โ,๒9\แ90U'ฟl	Vฟ?าrฟ?$ฟ;l}ฟงtpฟS	[ฟkท?ฟฏ ฟๅw?พๅผพ <uพผํฝ*?:vแ=\๚V>ฑ.>ํhฤ>4Z๊>Y!?x6?!"?qb.?ศ8?ํ๋A?i๏I?W๓P?1W?ซo\?ฤa?๐+e?Dถh?งสk?xn?หp?ฟะr?ภt?lv?iw?ฝx?	y?iz?ฅ({?Nฮ{?แ]|?Kฺ|?F}?~ฃ}?i๔}?:~?;w~?ีซ~?dู~?? ?	#?ฃ@?GZ?{p?ถ?^?สข?Hฏ?บ?wร?ห?า?ฐุ?๕??โ?xๆ?ไ้??์?l๏?ฅ๑?๓?<๕?ญ๖?ํ๗?๙?๒๙?ย๚?v๛???????r??ส????X????ย??์????1??M??e??z??????ฉ??ต??ฟ??ว??ฯ??ึ?????เ??ไ??่??๋??๎??๑??๓??๔??๖??๗??๘??๙??๚??๛??????????????????????????|uฟ๊ฟ[~mฟษFฟd5ฟฦcฎพคIืฝLe๕=ึฺฅ> U ?=&?[D?เ[?oฆl?ทw?Iๅ}???Kผ~?่ะz?ัt?๘?m?ํd?l
[?๘Q?ซสF?Kv<?142?	!(?TR?บื?'ผ?ฟ?Dw๕>.นๅ>$ำึ>Fยศ>ป>ธ
ฏ>	Vฃ>T[>๎>๖p>฿v>v	f>๙OV>ภกG>๎9>?&-><!>ๅ>๘ฤ>ผ>B๒=่แ=d้ั=Lbร=!?ต=Bฉ=`=n=s=$๛}=3^l=K๙[=๕ถL=,>=IK1=่?$=ุ=?โ=Q๘=n{๗<?Mๆ<กQึ<3qว<น<ฯถฌ<yน?<(</<<<q<S`<6ภP<%BB<กล4<ษ8(<๛<ปฌ<ข<L?;dศ๊;{ฺ;^Pห;ฬ2ฝ;ฐ;?ึฃ;แv;แ;a;นu;'ชd;?ษT;๚F;D8;vy+;ย;ฅ};q.
;x ;R๏:sด?:1>ฯ:ภฺภ:wณ:Pง:
i:ู:m:๏xz:&i:qๆX:[ืI:๋ำ;:ษ.:๖ฆ":\:?ู:,<>ฏsฝข!ฟพR!ฟ?#Rฟ_ฑpฟๅ~ฟึ'~ฟ@2rฟฎ]ฟ]ฎBฟ๐#ฟDฟpEรพผ/พ๋พิr)ผMห=M>yช>Vภ>ดๆ>ใ?ฌฤ?๑L!?ึA-?ซส7?A?,I?zHP?๏V?oํ[?@ฉ`?ษd?>`h?ูk?7n?p?ตr?/gt?|๒u?uIw?๓rx?๐ty?Tz?{?พ{?GP|?ฮ|?โ;}?ฅ}?ฟ์}?฿3~?{q~?ฺฆ~?ี~??~?ฬ?ี=?ูW?an?ไ?ส?lก?ฎ?น?ย?ฯส?๐ั?ุ?u??โ?ๆ?้?์?.๏?o๑?c๓?๕?๖?ฯ๗?่๘??๙?ฎ๚?e๛???????i??ย????Q????ฝ??่????.??J??c??x??????จ??ณ??พ??ว??ฮ??ี?????เ??ไ??่??๋??๎??๐??๒??๔??๖??๗??๘??๙??๚??๛?????????????????พ฿$ฟ๖ชbฟ~ฟข6zฟx]ฟ`/ฟkT๎พtๅkพฎฉ๋:9A_>า>W๓?ฤR5?เPP?Bud?Mxr?๙'{??T?ดฦ?ิ3}?้>x?#uq?ลNi?0`?mV?4IL?ฝ๙A?อฉ7?}z-?๓#?ั??i?ว?"7?>z๎>Pฌ?>'*ะ>{ย>[ต>?}ฉ>!>z>X>ู*>vแn>w^>ขWO>o A>??3>~'>B๕>ฤ4>/>?ฒ๛=ยM๊=มฺ=ห=๕ผ=(฿ฏ=7ฐฃ=xX=>ษ=ำ๔=ัu=d=AนT=u๗E=Y;8=แr+=9=ถz=ฟ,
=น =๚Q๏<~ต?<
@ฯ<,?ภ<?yณ<eง<Fl<'ข<พ<}z<i<ฉ์X<X?I<งู;<ฯ.<'ฌ"<๘`<?<ฯ<?๙๓;	ใ;iFำ;Yฤ;๛๔ถ;7Aช;5o;Io;?2;ปX;Um;];ํฤM;ฐ{?;\02;Qั%;'N;;v;ฑ๘:ไl็:ฅ[ื:๘gศ:~บ:iญ:้~ก:H:}ู:?#:?5r:ณda:ฉพQ:ภ.C:ฉก5:l):ธอu?ืC?็๘ํ>/๚=}Xพ}f ฟ}:ฟ๋bฟSyฟๅ?ฟtืyฟ`yiฟTQฟท4ฟ๘หฟํ็พo?คพAFพถูฝ?<+=๘>#z>ีช>ฟา>4/๗>oุ?uH?ณ'?V2?&B<?{๘D?L?2IS?OY?7^?ฉงb?f?๎โi?ะl?8[o?โq?|s?u&u?sv?lูw?ฮ๏x?3แy?zฒz?ใg{?|?a|?s}?มi}?cย}?-~?ฒQ~?Q~?:ฝ~?v่~?่?V.?mJ?มb?ำw??เ?ง?iณ?ญฝ?ฦ?Cฮ?ํิ?ณฺ?ณ฿?ไ?ว็?๋?ึํ?E๐?a๒?4๔?ษ๕?'๗?W๘?^๙?A๚?๛?ฑ๛?E??ล??4????็??/??m??ฃ??า??๚????<??V??m??????ก??ญ??ธ??ย??ส??ั??ุ?????โ??ๆ??้??์??๏??๑??๓??๕??๖??๘??๙??๚??๛??๛????????F0(?๕Cj>ฆๆ`พZฟ๘๛[ฟ44|ฟTs|ฟ่หbฟnถ6ฟภ?พจ,พ๒๙ผhA>l?ฤ>\q?4ไ0?o่L?้๛a?๖ัp?ฮ6z?ง๛~?ถ้?\บ}?#y?Dr?์j?a?uุW?hพM?=rC?% 9?|๊.?s๋$?6?ดู???LK ?/F๐><ฮเ>ฌ-า>฿`ฤ>Vbท>5+ซ>ฐณ>`๓>แ>.u>๛Jq>Pำ`>4rQ>C>?ฑ5>U3)>ๆ>ถฐ>จ>บF?=Sด์=*V?=ฃอ=hๅพ=\ญฑ=^ฅ=้=2>=?O=$#x=ถํf=3้V= H=h :=R6-=`1!=ผ=ง=d่=Tศ๑< แ<๕aั<6ูย<ฆRต<^ผจ<ท<-<R๚<k}<ฎk<'([<*๑K<ศ=<0<บX$<ห๏<ขQ<-p<ง|๖;ว_ๅ;sี;Vกฦ;ืธ;วฌ;ก?;น๓;U;พ?;ap;e_;ใO;.tA;ิ4;1';ฒไ;์;?;>@๛:?ฮ้:ู:๚wส:diผ:ฅTฏ:f(ฃ:ิ:๒I:พz:ฤณt:ถc:E็S:?0E:ฝ?@?๑5y?ำฟy?b?L?W๎?ฐ/>ีํ)พ6{ํพ`O3ฟs๘]ฟผ๊vฟ}แฟ!h{ฟRlฟ?`Uฟ!9ฟปtฟ<๐พtฏญพWzXพ\^ถฝธคี<ใ>[l>น๘ฃ>ฬ9อ>๗>๒>คฅ	?aU?V%?Zั0?m์:?ณฬC?K?ฉcR?โVX?M]?Vb?ๅ f?oi?ลkl?o?,Eq?Z:s?iํt?๔fv?}ฎw?สx?๊ภy?}z?กO{?๐{?({|?ช๓|?\}?ถ}?้~?อH~?~?ถ~?ฎโ~?ๆ?*?ฌF?_?u?ข?ร?ปฅ?ิฑ?Mผ?`ล?;อ?	ิ?๎ู?฿?sใ?G็?๊?vํ?๒๏?๒?๖๓?๕?๙๖?.๘?;๙?#๚?์๚?๛?1??ด??%???????%??e????ฬ??๕????8??S??j??~??????ฌ??ท??ม??ษ??ั??ื?????แ??ๅ??้??์??๏??๑??๓??๕??๖??๘??๙??๚??๛??๛??F}?๔ืj?|ฯ?Wป&>zพ%ฟEมbฟ~ฟำ-zฟ e]ฟ`F/ฟ๎พ
lkพ$?;จช_>โ/า>}?!b5?ฆ\P?ร}d?ํ}r?!+{?V?ฦ?ใ1}?๋;x?Zqq?hJi?ล+`?|hV?DL?๔A?ฆค7?mu-?#?ื?ี?f?ฬ.?>?ํ>าค?>#ะ>itย>ต>xฉ>>๐t>{>R&>?ุn>^>@PO>A>ึ3>x'>ซ๏>/>ภ*>๓ฉ๛=VE๊=้ฺ=@๛ส=4๎ผ=าุฏ=Qชฃ=๚R=!ฤ=๐=๖u=฿d=ฑT=P๐E=ณ48=ฑl+=w=Zu=ย'
= =VI๏<tญ?<8ฯ<6ึภ<bsณ<]?ฆ<ชf<๎<โ<rvz<%i<ำไX<ึI<฿า;<ตศ.<Gฆ"<[<iู<<๑๓;kใ;ว>ำ;?ฤ;`๎ถ;;ช;|i;๖i;่-;O;มm;
];ฝM;ฦt?;ํ)2;Tห%;H;k;ฃ;
จ๘:d็:?Sื:ผ`ศ:Owบ:%ญ:yก:!C:pิ:*:เ,r:\a:๖?พฯห>XคL?n|?3พu??ตC?ีฃํ>J3๙= Yพ๗ ฟE:ฟขbฟ$yฟึ?ฟัyฟoiฟๆQฟง4ฟ}ปฟHไๆพ5คพFพ`ฝฯ,=ข,>Nz>R1ช>jาา>z@๗>เ?FO?น'?_[2?ะF<??D?งL?TLS?"Y?9^?ฝฉb?Rf?ไi?sัl?h\o?๊q?๚|s?<'u? v?ฺw?P๐x?ฃแy??ฒz?8h{?h|??|?ช}?๑i}?ย}?Q~?ัQ~?l~?Rฝ~?่~?๙?f.?zJ?ฬb??w??็?ง?oณ?ฒฝ?ฦ?Fฮ?๐ิ?ถฺ?ต฿?	ไ?ษ็?๋?ืํ?F๐?b๒?5๔?ส๕?(๗?X๘?^๙?B๚?๛?ฒ๛?E??ล??4????็??/??m??ฃ??า??๛????<??V??m??????ก??ญ??ธ??ย??ส??า??ุ?????โ??ๆ??้??์??๏??๑??๓??๕??๖??๘??๙??๚??2ำ>^^?แ?CS?ต?ๅ>๎1ปูพf?ฟFpฟร?ฟ6rฟpANฟญฟฆฝยพ5๏พ&5ช=๊>`๒>H` ?ม@?กmX?>j?7v?D)}?/ใ?ญ&?^ฃ{?๔u?"n?f?jฏ\?ฬมR?H?4>?ฉ์3?ๆฯ)?f๔?นj?>?w?4๘>๒Q่>Gู>ห>ฑฏฝ>Aฑ>]Bฅ>เ(>.ย>l>Zำy>.ฬh>๕ใX>ฬ	J>d-<>?/>?/#>?๑>:w>\ณ>|4๕=#Aไ=iwิ=Yรล=?ธ=Rซ=๘s=g==#=u@o=tจ^=6O=rึ@=Pu3='=นj=ก=๓=	๚<๏้<๏ุ<ตเษ<F?ป<ศาฎ<๙ฏข<ศd<@โ<x<๒ t<kc<๊LS<กD<?๚6<วF*<tt<:t<7	<ua?;ฆํ;ฐ&?;ฬอ;dฟ;?6ฒ;+ืฅ;S;ณ;7ค;ในx;(ug;YcW;&oH;ป:;ค-;ถ!;๕M;?;(;X>๒:าlแ:8ฦั:ษ5ร:6จต:ฉ:?N:c:>9:!}:ี?iฟพ๋?พฌูv=4ุ?ธd?ย??ขภg?ํ*?๎ฎ>:ฟ0ปqมฅพษฃฟ:KฟgยlฟP}ฟFฟขๆtฟ~aฟษGฟ?P)ฟบฟฮพู3พ?พ:ๆ๒ผฦฅฆ=
9<>%>Frน>ำเ>?พ?Q?ี!??W+?6?r??OเG?-&O?IU?[?K่_?!d??อg?ง k?vศm?3p?MLr?ษt?ฉณu?๔w?ฎCx?๓Ky?1z?ป๗z?้ฃ{?$9|?uบ|?*}?}?ต฿}?(~?ฒg~?`~?ผอ~?ย๖~?J?9?ทS?อj?ห~???ฌ?Uท?ม?ษ?ฮะ?"ื???Zแ?vๅ?้?์?ฤ๎?๑?๓?ฯ๔?O๖?๗?ป๘?ต๙?๚?H๛?๊๛?v??๏??Y??ด????G????ต??แ????)??F??_??t??????ฅ??ฒ??ผ??ล??อ??ิ??ฺ??฿??ใ??็??๋??ํ??๐??๒??๔??๖??๗??๘??๘DฟA๓=yP1?3ซz?ณp?^๖#?HV>ฑ?sพl้ฟว^ฟื|ฟถำ{ฟm8aฟ๗4ฟ^๚พ0๓พpฉผvLJ>ฆเศ>๖ ?8@2?ด๔M?Nฟb?๒Tq?kz?ะ?วเ?็}?าำx?ี2r?;*j? a?คjW?^MM? C?ฎ8?ฮz.?~$?:อ?ut?บ}?๛??>ไ๏>G(เ>๐ั>)อร>`ืถ>ฎจช>>9>ฃ>v>ต>'p>ู#`>aฮP>(~B>:#5>Rฎ(>ไ>!=>๓%>~?=]๙๋=จ?=vuฬ=dNพ=ฟ ฑ=ก?ค=7o=ผฬ=`ๆ=^w=ฐ6f=ู>V=bG=ึ9=๛ฌ,=ฑ =ฦ=๐)=\=๑<ฅMเ<฿ปะ<ฆ>ย<ะยด<6จ<&<Bซ<r<ฆJ|<ูฦj<IzZ<^OK<?1=<t0<Wึ#<uv<นเ<<น๕;ฬฉไ;ฒษิ;ฟฦ;\Dธ;Pyซ;ฃ;};].;h;๊Qo;^ด^; >O;ณฺ@; w3;G';i;ำ;i;็x๚: ้:lๆุ:Wืษ:่ำป:ษฎ:๓ฆข:\:ูู:dอVฟ0~ฟfฆ8ฟ=ไOพ9?ฑ>D?Wz?(ฆx?FJ?ซ?>?e >8#8พ>h๓พ5ฟng_ฟ?wฟ๛๑ฟ`๔zฟ_ykฟ@Tฟฟ7ฟ๑ฟUซํพ-าชพ
๑RพF฿ซฝbุ?<ฅ>1ฒp>nืฅ>่ฮ>cภ๓>@Q
?ํ?)?%?๐G1?ฐT;?,(D?ชไK?ถฉR?X?จฝ]?ี=b?`)f?ฦi?dl?ฉo?I\q?jNs?ิ?t?vv?ปw?๐ีx?ลสy?	z?	W{?๖{?ธ|?|๘|?@`}?(บ}?~?K~?๗~?ธ~?rไ~?n
?S+?ัG?`??u?`?h?Jฆ?Oฒ?นผ?ฝล?อ?Oิ?*ฺ?<฿??ใ?n็?น๊?ํ?๐?/๒?	๔?ฃ๕?๗?;๘?E๙?,๚?๔๚?ก๛?7??น??*????฿??(??h????ฮ??๗????9??T??k????????ฌ??ท??ม??ส??ั??ื?????แ??ๅ??้??์??๏??๑??๓??๕??๖??\?ฟDฏ9ฟ8&ฮฝY?+Ml?P๖{?B?ศPถ>?8วฝ็บ ฟ0ฏLฟbvฟ๖Mฟฯ5lฟIDฟnหฟ!Aฉพรฝํ>>ช>น?5ฎ'?UภE?gท\?;m?ดx?ไ~???m~?z?t?~ๆl?*&d??Z?qP?qZF?&<?ล1?฿ด'?u้?ณr?X[?cช?ฏว๔>อๅ>ไ5ึ>๛-ศ>
๖บ>ฎ>๎ฺข>๏็>?ฅ>?>๖!v>ำXe>ซU>หG>๕^9>ก,>5ฟ >uซ>vX>กน>6๑=rำเ=๛Eั= สย=oMต=กพจ==+=!	=๛4}=ภฅk=M[=*L=s๎==?ภ0=}$=๓=ks=z='บ๖< ๅ<=ชี<mีฦ<น<็/ฌ<๎;?<S<\ย<T <๎Rp<หฃ_<$P<eชA<j84<_ต'<ฑ<๎:<ผ%<x๛;๚๊;ฺะู;ฑส;?ผ;ฏ;เVฃ;ฦ?;:r;=?;๙t;๗c;ภ#T;IiE;ด7;๓*;;ฅ	;~ย	;2 ;๎:x?:Kฮ:Dภ:ุ๊ฒ:ู~ฆ:ข๏:r;;0ฟ%ณ~ฟะำXฟ?๓ฤพฎ'5>'?Y8o?0ษ~?G]?ม?+บ>g?ฝ๕cลพ"$ฟทฬSฟqฟ๔ี~ฟซ?}ฟัyqฟz\ฟษpAฟ
"ฟfฑฟoภพู|พ?๛ฝ๊หดปnsิ=.8Q>>วย>ภ=่>j0?`?฿ึ!?ป-?ป58?kA?u~I?P?เพV?[$\? ู`?ฦ๒d?h?fk?pRn?ไชp?dดr?#yt?v?๙Vw?ฌ~x?y?o]z?,{?;ล{?V|?zำ|?0@}?a}?๛๏}?ฌ6~?่s~?๓จ~?ๆึ~?ฒ?~?*!???฿X?Do?จ?t? ข?ฎ?น?๔ย?"ห?8า?[ุ?ซ??Eโ?Aๆ?ด้?ฑ์?H๏?๑?v๓?%๕?๖??๗?๓๘?ๅ๙?ถ๚?l๛?	??????m??ล????T????ฟ??๊????0??L??d??y??????จ??ด??พ??ว??ฯ??ี?????เ??ไ??่??๋??๎??๐??๒??๔??ุ\	ฟํ@|ฟจดRฟ,ดxพยำ>เ)^??M๏R?3ๅ>ฉkปUโูพm?ฟ>จpฟก?ฟx)rฟ+Nฟ&ฟAยพ๔vพซ=@R>ก๒>
r ?ฦ@?*xX?{Ej?ต;v??+}?ใ?%?ก{?ผ๐u?n?f?ช\?ธผR?๑H?ไ.>?็3?แส)?๏?f?:?7s?e,๘>/J่>A@ู>"ห>,ฉฝ> ฑ><ฅ>|#>!ฝ>ฒ >สy>์รh>??X>J>ฌ&<>ึ8/>'*#>j์>'r>กฎ>ฌ+๕=๎8ไ=ฤoิ=:ผล=ธ=iLซ=9n=ปa===ิ7o=l?^=/O=}ฯ@=ึn3=๛&=e=b==?w๚<้<ส็ุ<kูษ<~ึป<xฬฎ<ชข<P_<*?<ผ<"๘s<8c<IES<qD<@๔6<ก@*<ผn<็n<2	<<X?;๏ํ;ด?;ซฤอ;z{ฟ;10ฒ;.ัฅ;N;;d;่ฐx;อlg;[W;้gH;?}:;_-;แ~!;H;wู;ู#;5๒:ฎdแ:คพั:ผ.ร:ฆกต:jฉ:ะX?r.>dฟณUxฟiฟie?พFX{=w?ิd???ชฉg?S?)?&ฎ>ฆH\ปฆพ'ยฟUOKฟะฮlฟโT}ฟ๋ฟ๗?tฟๆsaฟ{Gฟ๗@)ฟ๑ฉฟmฮพ?พิพ ๑ผง=ik<>>๕น>bเ>Fว?้X?Y(?ฝ]+? 6?ไ??5ไG?)O?EU?ฑ[?๊_? #d?ตฯg?%k?รษm?'4p?HMr?ฃt?fดu?w?<Dx?nLy?1z?๘z?:ค{?i9|?ฑบ|?ท*}?ฦ}??฿}?ต(~?ะg~?y~?าอ~?ี๖~?[?9?ฤS?ืj?ิ~?#? ?ฌ?Zท?ม?ษ?าะ?%ื???\แ?xๅ?้?์?ล๎?๑?๓?ะ๔?O๖?๗?ผ๘?ต๙?๚?H๛?๊๛?v??๐??Y??ด????G????ต??โ????)??F??_??u??????ฅ??ฒ??ผ??ล??อ??ิ??ฺ??฿??ใ??็??๋??ํ??๐??๒??" ื>d็พ(Dwฟ5^ฟ+ํขพ?ฐ>U?ใ??[?OE?>B=ฟฉรพ@ุ7ฟK?lฟ&ุฟZ๖tฟLSฟฎม!ฟว ะพ^r.พ็ปp=o๗>~ห่>l?แ<?ญV?ซh?'u?ฃ|?	ล?la?$|?ภซv?~o?Jg??ภ]?6฿S?FซI?X??5?๊๊*?L!?๏s?๘<?j?๋๚>^๊>4ๆฺ>@ฬ> ฟ>qฒ>zฆ>ถY>ด฿>>ว{>%j>Z>็?K>Bฉ=>lก0>az$>ุ%>C>ะพ>ั&๗=Sๆ=ผ'ึ=๔Uว=#น=ฃฏฌ=๒ธ?==97=L=(q=ธn`=Y?P=๙_B=ใ4=iV(=๗ง=ฺศ=นช=w?<ใ๙๊<ชฺ<่|ห<฿\ฝ<ย7ฐ<-?ฃ<ํ<๑<0&<9๓u<เd<k?T<3F<p8<ข+<ธ<[ก<ปO
<ต <ู๏;S๊?;bpฯ;	ม;ขณ;ฺ)ง;ษ;?ม;%ต;ุตz;ูMi;9Y;{J;ฅ<;๔.;ฮ";๋;*?;]2;-๔:19ใ:?rำ:sฤฤ:2ท:oNh?l]d?><?พTฑrฟ{7pฟ?ฟ
K๕บpe?3^?jถ?ธl?.&2?อ5ย>Uั=ธพูฟนkFฟ"์iฟB|ฟทฟาvฟ
dฟงJฟ่ฬ,ฟีdฟ๔ีพrพ ๔%พ&ถ-ฝfb=11>๕>e?ด>j?>ๆา?>+ฐ?4ฐ?ฏ*?G?4?>?G?dN?ูT?n|Z?g_?ใฐc?clg?รซj?ซ~m?๊๒o?กr?u๎s?ธu?๎v?$x?0y?\z?*ใz?{?ฑ)|?ญ|?้}?}? ื}?!~?*a~?ท~?ีศ~?๒~??แ5?๕P?ih?น|?P??พช?+ถ?ภ?กศ?ะ?zึ????เ?	ๅ?ฆ่?ว๋?}๎?ึ๐??๒?ก๔?'๖?y๗?๘?๙?v๚?4๛?ู๛?g??ใ??N??ช??๚??@??|??ฐ???????&??C??\??r??????ค??ฐ??ป??ฤ??ฬ??ำ??ู?????ใ??็??๊??ํ??๐??r}?vไ>?h?พํuฟaฟคญพฌง>๎cR?l๔?฿)]??`?พิy=kคฝพRบ5ฟhษkฟZฝฟฒจuฟZTฟ๐b#ฟัำพๅค5พU=๊>+ๆ>ืU?<?jU??h?X?t?u|?๛บ?ัo?oF|??v?ปบo?Tg?
^?Z,T?๚I?(ง??๚[5?ต7+?ํQ!?๗ป??ฌ?๚>๊>๕V?>ฆอ>Gฟ>ษฯฒ>๎฿ฆ>จฌ>f->TY>O|>;k>ล[>ดL>ฆ>>?1>Oิ$>ญy>dไ>>xฎ๗=ฎๆ=kึ=รว=0๏น=จญ=hก=๙็=ไ=ฉื=]ญq=b๊`=rPQ=หB=FG5=4ณ(=T?=9=๕=ฐ?<t{๋<"#?<ํห<Lลฝ<๐ฐ<Vค<๎<BP<o<?zv<V^e<โqU<g?F<Aึ8<,,<$ <U๓<
<|? <๗๐;Fe฿;หโฯ;๗sม;งด;ง;ไ;ิ;q?; @{;ฮi;๗Y;๊wJ;Wi<;T/;^(#;{ิ;ํI;นz;ฝณ๔:ถใ:@็ำ:๚0ล:ื>๖3e?rMg?ื>H$๔พคํpฟๆqฟQูฟญ๙ผัํ ?๐O\?๛?ตสm??N4?ผgว>ญ8=พ๙ฟฒEฟ*iฟฎ๐{ฟฺฆฟร๓vฟvฌdฟฉ|Kฟ1ฝ-ฟrbฟ,๕ืพjพ?น)พ๏;ฝฦ=?.>ผ>ฃณ>์J?>xา?>+>?8K?ธ)?รญ4?"Q>??ฦF?ง/N?ูชT?TZ?YD_?8c?ฑQg?j?|jm?aแo?fr?<แs??~u?ไv?}x?)y??z??z?5{?w%|?hฉ|?ผ}?ฬ~}?ิ}?๘~?`_~?+~?~ว~?Z๑~??5?4P?ยg?(|?ำ??_ช?ูต?ษฟ?dศ?ุฯ?Lึ?ใ??บเ?๋ไ?่?ฑ๋?j๎?ล๐?ะ๒?๔?๖?p๗?๘?๙?p๚?/๛?ิ๛?c??฿??K??จ??๘??>??z??ฎ???????%??B??\??r??????ค??ฐ??ป??ฤ??ฬ??ำ??ู?????ใ??็??๊??ํ??Dy&?ดํ{?8า>?๋พ\?wฟB]ฟ4พด>๙U?W??=QZ? ั๛>?-=n๋ลพeข8ฟDmฟuเฟอฑtฟvRฟ$!ฟงฮพนผ+พะ๒z=ษ>้ว้>rิ?Q4=?APV?ภh?Du?~ฌ|?ศ?ี[?พ|??v?฿go?๖f?	ฅ]?#ยS?I?A:??๐4?ฮ*?4๋ ?ำX?๔"?ฉQ?ำ๙>?้>ศปฺ>8rฬ>_๚พ> Nฒ>3fฆ>:>zย>D๕>็{>Woj>mkZ>9wK>]=>%}0>X$>O>฿x>nฃ>ห๓๖=ฬแๅ=x๛ี=ป,ว=ภbน=ๅฌ=ซ?=ฃv=c=tu=๖p=5@`=ฒP=ญ7B=พ4=3(={=ช==L?<'ษ๊<3}ฺ<ณRห<5ฝ<5ฐ<)ฺฃ<Fz<|ไ<ว
<3ภu<กฐd<;ะT<๙	F<EJ8<็~+<๊<<
3
<ฬ <&Z๏;ผ?;YEฯ;sแภ;O}ณ;,ง;n;๕ฃ;2;ีz;si;.๎X;?I;คฺ;;ะฯ.;หฌ";|a;๊?;%;h๚๓:
ใ:ยFำ:่zBฟโๆ5>9mi?@8c?!> ฟ%TsฟCoฟฌฟ๑I;%ฑ??ๅ^?{ล?vl?ฅT1???ภ>๗"?<Fyพnฟ?๋Fฟฑ8jฟภ_|ฟใฟ^vฟpมcฟ[VJฟ>r,ฟ=ฟำ2ีพ๚ดพ$พ~\(ฝช?=x92>คz>Vต>๏ี?>ข ???&ึ?03*?ฦ5?ฐ>?าG?wxN?|๊T?Z?ศt_?hผc?jvg?{ดj?@m?๙o?Yr?l๓s?u?O๒v?]'x?e3y?ฬz?Gๅz?๋{?G+|?rฎ|? }?}?ๅื}?ฮ!~?ีa~?L~?Vษ~?๓๒~???56?>Q?จh?๏|??ต?แช?Iถ?*ภ?ธศ?!ะ?ึ???้เ?ๅ?ฐ่?ะ๋?๎??๐?ไ๒?ฆ๔?+๖?|๗?ก๘?๙?y๚?6๛??๛?i??ไ??O??ซ??๛??@??|??ฐ???????&??C??]??r??????ค??ฐ??ป??ฤ??ฬ??ำ??ู?????ใ??็??๊??hพ)๗:?)่u?ผค>gบ	ฟqR|ฟ
~RฟAYwพ60ิ>LP^?	?JษR?งๅ>?ปDฺพฤจ?ฟ.ธpฟx?ฟRrฟฝNฟIuฟ?Dยพต?พไ๗ซ=>พถ๒>ษ ?ษ@?ฑX?แLj?a@v?๛-}?ไ?n$?ง{?lํu?n?	f?ฌฅ\?ฅทR?ล}H?ต)>?iโ3??ล)?จ๊?Ta?5?๊n?6$๘>mB่>๊8ู>6ห>จขฝ> ฑ>?6ฅ>>ธ>๘๛>ญมy>ชปh>ิX>f๛I>๕<>2/>P$#>๘ๆ>m>็ฉ>?"๕=น0ไ=hิ=ตล=?ธ==Fซ=zh=b\=ข=?=3/o=d^='O=ศ@=\h3={๕&=_=*=<=๔n๚<้<๖฿ุ<"าษ<ถฯป<)ฦฎ<:คข<ูY<ุ< <S๏s< c<จ=S<WD<ฅํ6<{:*<i<i<-	<O?;Zํ;ธ?;=ฝอ;tฟ;ม)ฒ;1หฅ;xH;T;;ํงx;rdg;หSW;ฌ`H;Cw:;-;y!;C;kิ;&;ฺ,๒:\แ:,)uฟ฿.ฟVY>mr?6หW?๔,>?ณฟYkxฟ๛hฟ฿?พกึ=ฎF?๐d?X??จg?ดถ)?0>ฎ>=่ปบZฆพเฟคdKฟ4?lฟ@Y}ฟฟJืtฟLhaฟ;mGฟ1)ฟBฟjKฮพ๒พ8พึ3๏ผNง=ฦ<>๔4>ขน>๏ชเ>ฯ?D`??.?}c+?ฎ%6?W??่G??,O??U?J[?ิ์_?๚$d?mัg?คk?หm?H5p?CNr?} t?#ตu?<w?สDx?้Ly?์1z?t๘z?ค{?ฏ9|?ํบ|?๋*}?๓}?เ}?ื(~?ํg~?~?้อ~?่๖~?k?-9?ะS?โj??~?+?'?"ฌ?_ท?ม?ษ?ีะ?(ื?ก??_แ?zๅ?้?์?ฦ๎?๑?๓?ะ๔?P๖?๗?ผ๘?ถ๙?๚?H๛?๊๛?v??๐??Y??ด????G????ต??โ????)??F??_??u??????ฅ??ฒ??ผ??ล??อ??ิ??ฺ??฿??ใ??็??%vฟEiไฝ!U?ฑูg?ซ+9>*%ฟาภฟ@ฟmุ	พด ??Hi?D}?z:F?ใญม>t๊ฝษB๘พชฐIฟึJuฟHฟถmฟใFฟซฟhFฏพฦฺฝ(๒=๕"ฅ>พ ??%?/mD?Cป[?l?งw?ฟ?}?่๛?Rม~?]ฺz?[?t?YOm?d?ว[?Q??F?ฃ<?JG2?ณ3(?nd?+้??ฬ?ฑ?๕>๊ีๅ>M๎ึ>ใ?ศ>ปป>a!ฏ>Mkฃ>Co>$>m>ซ?v>๚'f>xlV>YผG>j:>#>->ณQ!>4>ทื>30>!c๒=8กแ=า=|ร=๓ต=ZYฉ==0ฑ=๕=`~=~l=๕\=าL=?>=3c1=*%==F๖=B
=า๗<mๆ<nึ<ว<ฒน<ฮฌ<*ฯ?<Wฅ<?A<?<ล/q<Nq`<b?P<]\B<?4<}O(<?<dภ<๎ก<n?;่๊;ฺ;ฯkห;VLฝ;?'ฐ;๛์ฃ;u;6๔;3;ดฺu;ษd;ดๆT;ดF;k]8;+;Kง;ฐ;A
;ำง ;Rr๏:6โพ"g~ฟซ2ฟู>ว{?1CC?หฯ3=:J)ฟ~ซ}ฟ-]ฟ๙?าพUQ>?!?๚l?ฉF?[ไ_?Oฉ?a>งiฝ๋พพ@.!ฟใูQฟ4pฟ~ฟD4~ฟงQrฟ๓ฏ]ฟ๔ไBฟ-,$ฟาZฟมยรพัฉพะ๏พ*7ผอธษ=NL>BV>ค
ภ>pๆ>b?ตฉ?5!?ศ,-?%ธ7?ภ?@?ุI?<P?
uV?๎ใ[?๗?`??มd?๗Yh?czk?E2n?๒p? r?dt?ส๏u?Gw?์px?.sy?Sz?6{?wฝ{?IO|?ฅอ|?#;}? }?0์}?b3~?q~?|ฆ~?ริ~?ุ?~??ก=?ซW?9n?ย?ฌ?Sก?ฎ? น?ย?มส?ไั?ุ?l??โ?ๆ?้?์?)๏?k๑?_๓?๕?๖?อ๗?ๆ๘?ฺ๙?ญ๚?c๛???????h??ม????Q????ฝ??่????.??J??c??x??????ง??ณ??พ??ว??ฮ??ี?????เ??ไ??ณ@@ฟษ]ฟฯf๖=u?n?zzM?รฑปฆๆDฟk6~ฟEg$ฟฟฐ;7ิ?Yt?์*w?ท4?์a>h)พลจฟUฟTzฟ๑?}ฟ0gฟr =ฟ๗ืฟIcพใสrฝM&>
น>;t?สฺ,?ศI?ฒ_?ฯEo?Ny?P~?<๛?&~?Fฦy?tms?ะk?#ตb?Y?๔O?nพD?k:?/00?)&??i??บ๙	?'Y?หF๒>ฉณโ>4๘ำ>ฦ>?๘ธ>;ฉฌ>ก>BC>๘>X>๐ps>ีb>๋QS>มึD>S7>?ธ*>๘>5>อ	>BI >โื๎=,T?=๒ฮ=ตภ=2Iณ=๛?ฆ=N==X=cz=ฤi=&?X=ัI=?ะ;=ศ.=ยง"=)^=๋?=+=๚๓<ึ
ใ<jHำ<๊ฤ<๐๗ถ<lDช<r<ณr<G6<x_<๊คm<f%]<หM<?<๓52<?ึ%<-S<Y<๎ฃ<nน๘;สt็;cื;่nศ;บ;|ญ;ก;?M;o?;{(;=>r;ผla;&ฦQ;ป5C;*จ5;|);๖N;	c;99;?:
)?ญฟ๒#~ฟ<0ธพ๕ฐ?
??l#?๑ฝผ;Dฟ?ฟVJฟตพ,T>๎5?mPu?ซv|?๋T?ดB?
<\>! พผไ?พซ,ฟ?Yฟ.ตtฟฤฟ%|ฟณnฟ
Xฟาิ<ฟฤฟื๗๘พ
ถพชงhพdีฝ]E<ฝ๕=?s`>	{>BHศ>าํ>ญ?ฐ?วส#?dt/?น9?kฟB?ฒจJ?fQ?ฟขW?*๋\?fa?ฉe?ัi?l?ชตn?q?;?r?บt?n:v??w?ฉx?฿ฃy?P}z?ฮ9{?.?{?ลj|?vๅ|?ลO}?แซ}?ญ๛}?อ@~?ฏ|~?ฐ~?{?~?e?&?KC?\?yr?p???ค?gฐ?ป?Nฤ?Nฬ?<ำ?<ู?n??๎โ?ำๆ?3๊?ํ?ง๏?ุ๑?พ๓?b๕?ฮ๖?
๘?๙?๚?ิ๚?๛???ฅ????{??า????]????ฦ??๐????4??P??g??|??????ช??ถ??ภ??ศ??ะ??ึ?????แ??iy>ฑkฟู?-ฟฮ๒ล>๗m~?ัm#?โgพบYbฟRถsฟจx?พสจ/>ุA:?อ|?
?k?B๘?	ฏ0>A๖พKY#ฟหอaฟL?}ฟ่zฟ6^ฟ#b0ฟk๐พ^pพแย'ปE3[>>ะ>ฉ6?Vป4?ห?O?X!d?น@r?ฏ{?
J?_ฬ?ศF}?8\x?Hq?yi?S_`?ศV?ฯ{L?พ,B?y?7?Dฌ-?lต#?ญ
?pน?ืส?*?>VS๎>๖?>ปoะ>ขผย>ืต>็ทฉ>bW>ญ>?ฏ>iW>ร4o>Fเ^>G?O>:dA>94>น'>=,>h>Z_>๘?=ค?๊=๔fฺ=sJห=๔7ฝ=~ฐ=@๊ฃ==๛=ค#=๛๓u=?ๅd=ลU=ฝ=F=ร|8=ยฏ+=โล=rฏ=า]
=fร =?ฆ๏<฿<ฉฯ<ฐ!ม<นณ<บ?ง<~ฃ<ี<ว<|ุz<ani<ป9Y<%J<e<< /<๓ๅ"<ม<<bE<P๔;HZใ;zำ;4แฤ;?5ท;ต}ช;ง;ซฃ;c;uณ;ร๒m;m];
N;นฟ?;ฌo2;;&;๚;ส;๐ฮ;ฟ}?\`ศ>ฦ<ฟ๊lฟTโฝญE?/^y?i-๏>ธพ}m^ฟ4|ฟ?/ฟ?_!พึrฦ>F K?)|?dv?ฑE?8<๑>O>f<Rพ๊8?พ9ฟซ๘aฟํีxฟษ?ฟ฿zฟc?iฟ#RฟrN5ฟ๘mฟณP่พ6ฅพญทHพฝ"=๘>ใKx>\Hฉ>๏ า>'๖>ภ?\?iิ&?ำ!2?;<?3ะD?ฮwL?\*S?cY?ฃ^?4b?ณsf?pำi?ยl?Oo?ทq?Ass?ฬu?ฮv?ฉำw?ฮ๊x???y?ธฎz?กd{?L|?๎|?T}?๋g}?ฬภ}?ฬ~?P~?H~?Uผ~?ฏ็~?<?ม-?์I?Qb?rw?พ??Pง?3ณ?~ฝ?gฦ?ฮ?ฯิ?ฺ?฿?๓ใ?ถ็?๗๊?ษํ?:๐?W๒?,๔?ย๕?!๗?R๘?Y๙?=๚?๛?ฎ๛?B??ร??2????ๅ??.??l??ข??ั??๚????;??V??m?????????ญ??ธ??ย??ส??ั??ุ?????ศถi?Aqพ0สฟ๊หพโ&?จึ{?๙\ั>ฬก์พK๖wฟp๎\ฟ?jพะ"ต>E$V?๙??๒-Z?ญc๛>;g)=วOฦพuล8ฟ9Vmฟอแฟฬฅtฟ?Rฟ!ฟฺjฮพD+พน|=ษO>ศ๓้>ๆ?ฒB=?[V?ทวh?๓Hu?ฏ|?:ษ?ูZ?|?v?๊co?
๒f?1?]?ฝS?kI?5??๛๊4?๙ศ*?Qๆ ?T?m?VM?fห๙>ถำ้>gดฺ>Bkฬ>ะ๓พ>๖Gฒ>i`ฆ>5>eฝ>๐>{>gj>ฉcZ>๙oK>{=>ึv0>ฌR$>ำ >รs>ช>๋๊๖=ูๅ=ล๓ี=%ว=\น=ฎฌ=โ?=@q=_=ศp=ืํp=8`=ชP=ซ0B=ท4=o-(=ี=\ฅ=ฒ=?B?<ญภ๊<Puฺ<\Kห<ร.ฝ<ูฐ<>ิฃ<ฤt<\฿<<Tทu<_จd<ศT<าF<C8<ถx+<'<)}<.
<' <Q๏;	ด?;?=ฯ;|ฺภ;ิvณ;$ง;ๆh;ผ;V;ษxz;i;YๆX;GืI;?ำ;;ษ.;๋ฆ";\;ิู;"๐ะ>?ฦx?เ๓%=ฃๅjฟ? Bฟนโ7>mi?ฑc?x~>SฟpsฟซqoฟxHฟธท;ถ๊?_?ํว?hl?01?ฃ่ฟ>wจ๘<FวพฉฌฟGฟ๕Ejฟเd|ฟ#ฟcWvฟ-ถcฟKHJฟub,ฟ๔ฟ3ีพพSI$พHn'ฝQM=Kl2>ฦ>|kต>ฒ่?>" ?uโ?พ??9*?็5?ด>?ลG?์{N?ํT?<Z?w_?iพc?(xg? ถj?m?ฅ๚o?Xr?I๔s?วu?๕๒v?ํ'x?โ3y?9z?ฅๅz?<{?+|?ฏฎ|?O }?ย}?ุ}?๑!~?๓a~?f~?mษ~?๓~??C6?JQ?ณh?๙|??ผ?็ช?Nถ?/ภ?ผศ?$ะ?ึ???์เ?ๅ?ฑ่?ั๋?๎??๐?ๅ๒?ฆ๔?,๖?}๗?ก๘?๙?y๚?7๛??๛?i??ไ??O??ซ??๛??A??}??ฐ???????&??C??]??s??????ค??ฐ??ป??ฤ??ฬ??ำ??ู??/V?)p#?พำฟ๑oฟ(
#ฝ้^?tค`??=ุ/ฟด?ฟ้๊7ฟU:ปฝ~๖	?ุm?ส{?ณื@?z6ณ>3ำฝ๘ฟ์{MฟDๅvฟz7ฟ หkฟๆCฟฑฟงพชผฝฬ>ศRซ>จ?8#(?ัF?J๛\?ุim?N/x?~?{??ค~?งz?znt?ืษl?Ed?ิ}Z?*zP?ฃ6F?bโ;?:ข1?f'?
ศ?R?<?๗?ย๔>ฯ?ไ>ะึ>ม?ว>ษบ>ว]ฎ>ผณข>1ร>w>ซ๋>ลๅu> e>vU>วึF>:19>_v,>q >d>๊5>r>?J๑=เ=๕ั=ฏย=T ต=ขจ=sๆ=ฐ=>็=ๆ๕|=kk=๖[=LไK=ฟ==ฮ0=T$=ศ์=ๆO=lo=?|๖<?`ๅ<๔tี<ึฃฦ<ู๋ธ<๕ฌ<๗?<"๗<ภ< <?p<l_<;้O<zA<v4<'<ร้<ด<<นH๛;ึ้;ู;๚~ส;๒oผ;วZฏ;!.ฃ;?ู;๑N;g;xผt;งพc;ี๎S;8E;ล7;?ศ*;kํ;ทไ;?7ฟผ	E?wI?J{ฒพฬฟ้0?พำ๕>?	~?๕:?&4ฮป๊2ฟํ~ฟะคWฟ๊%มพ6ฐ<>-^(?7ฮo?ก~?ษ\?ฏ?T>W?ฝ4`วพมฬ$ฟpRTฟ?ไqฟR้~ฟร}ฟ'>qฟบ%\ฟAฟ#"ฟ=ฟwฟพi{พ๒)๘ฝCwปน\ื=uR>},>tย>์บ่>9h??ท"?0โ-?ยW8?เA?กI?mงP?แาV?ฯ5\?V่`? e?h?lฉk?'[n?vฒp?๖บr?ุ~t?v?D[w?ex?Uy?<`z? {?Vว{?ึW|?ี|?A}?}?๑}?7~?ญt~?ฉ~?zื~?2?~?!?d??2Y?o?็?ช?/ข?มฎ?คน?ร?=ห?Oา?oุ?ผ??Tโ?Nๆ?ภ้?ป์?P๏?๑?}๓?*๕?๖?เ๗?๗๘?่๙?น๚?n๛???????n??ฦ????U????ภ??๋????0??L??d??y??????จ??ด??พ??ว??ฯ??ี??ผฆ?.M>LfฟMฬ5ฟ4Lฒ>ข'}?]*?GFพก^ฟ?uฟฆฟ}r>ฃ!6?๓?{?Q๏m?'ฝ?แ5B>๖พ\ ฟ`ฟkh}ฟ*{ฟปข_ฟ-S2ฟว๕พpyพ6ห4ผNS>Tฮฬ>cฦ?k3?๙N?อ|c?fำq?ปสz?ภ3?ึ?์j}?x?โq?ฬi?๗น`?U?V?๛?L?บB?ู>8?๒.?$?ทe?๑??(?>ฐ๊๎>D฿> ๗ะ><ร>๋Nถ>(ช>ม>๚>ข>ฎ>พึo>w_>-P>่A>54>ฟ+(>(>ชห>8ผ>น?=ีA๋=?ฺ=Dึห=&บฝ=ธฐ=[ค=๗=[]=ณ~=v=e=กU=nฦF=?๛8=*&,=4 = =Dฝ
=9=PL๐<v฿<ุะ<๏ฆม<5ด< ณง<แ<x9<$<{<sj<ฯY<ฐJ<2<<๋/<[V#<[?<ไq<๗<#๙๔;+๗ใ;x#ิ;iล;iดท;[๓ช;๛;	;jย;๔1;๕n;c^;:N;
D@;อ๊2;ะ~&;๏;o?ฟูUฝ๔ฯz?|?>ผ<4ฟ?๙oฟq>พ??ณ'{?@?>ณพ๓Zฟั8}ฟๆ3ฟ?,7พh์ผ>yH?Z{?ั{w?-G?๗>Z>Fพ๑.๙พฌ7ฟ|ศ`ฟGxฟ?ฟ}zฟg?jฟ|!Sฟs6ฟจฟขา๊พไ จพ6Mพ~กฝ๚จ=&>พt>ทฌง>ทะ>	:๕>_๙
?ฎ?๎`&?ผ1?วบ;?ปD?3L?I๎R?๏ฯX?ใ๑]?Wkb??Pf??ตi?\จl?ด8o?็rq?bs?฿u?ฺv?mศw?แx?jิy?eงz?H^{?ห?{?*|?3?|?Wd}?ฒฝ}?~?-N~?D~?บ~?,ๆ~?์?,?๑H?xa?ตv??	?ึฆ?ษฒ?"ฝ?ฦ??อ?ิ?eฺ?o฿?อใ?็?ฺ๊?ฐํ?$๐?D๒?๔?ด๕?๗?G๘?P๙?5๚??๚?จ๛?=??พ??.????ใ??+??j?????ะ??๘????:??U??l?????????ญ??ธ??ม??ส??ั??็กXฟ< ?FE`?eทพเฟ.วซพฆ2?9รx?๑ท>dฟจzฟๆWฟจพ.๚ฦ>ี่Z?oะ?ทV?๙ผ๎>xฒx<มัพ๖ฝ<ฟOoฟ<?ฟก:sฟู Pฟ?ฟ~วพๆiพ?k=&&>ฝ๗๎>ช๘?ๆ>?ะW?งi?ฏืu?ฯ๘|?ฅู?ถ<?lา{?g6v?ํ๎n?ตmf?]?(S?a๐H?>?T4?5*?ฑV ?ษ?x?[ฮ?ศู๘>๎่>ถ?ู>อห>G3พ>๔ฑ>|ถฅ>ภ>'(>ุd>ฯz>๑ri>ณY>,J>
ต<>กฝ/>๙ฅ#>ึ_>น?>?>nๆ๕=ใๆไ=ศี=Sฦ=ธ=8ฯซ=่=ำ='=ฟ็=น๎o=จJ_=อO=๖bA=๘3=@{'= ?==๚=ฆ7๛<โว้<วู<ไsส<>fผ<?Rฏ<&ฃ<+ำ<๚H<z<?ฒt<?ตc<?ๆS<๋0E<I7<๐ย*< ่<ภ฿<	<ื <ฯS๎;๔ว?;+bฮ;ภ;ธฒ;Pฆ;'ฤ;l;ซ;Coy;๑h;i X;OI;พ;;6.;~๚!;ฟgฟฆUฟ่ใ๖>cs?e}?ผt)qฟ*e7ฟฤq>่n?งะ\?x!P>๑ฦ
ฟ๚nvฟเkฟZธฟญ=t?ewb?s๘?฿i?ภ๓,?Dูต> /<๛ฑพฤ;ฟ!Iฟฤkฟ๕|ฟ?JฟQuฟๅgbฟมจHฟ*ฟฮ

ฟA4ัพหพพ ฝ?=f?8>F8>Tะท>ษ฿>?ลผ??ณใ*?/ต5?8??~G?4แN?FU?|?Z?oบ_?๙c?8ซg?jโj?-ฎm?0p?y8r?t?นคu??w?p8x?3By?ฃ(z?h๐z?{?ฃ3|?ฐต|?a&}?}??}?ใ%~?_e~?\~??ห~??๕~?๛?๎7?ผR?๓i?~?w??ซ?๊ถ?ถภ?1ษ?ะ?ๆึ?h??-แ?Oๅ?ใ่??๋?ซ๎??๐? ๓?พ๔?A๖?๗?ฑ๘?ฌ๙?๚?A๛?ไ๛?q??๋??U??ฐ?? ??D????ณ??เ????(??E??^??t??????ฅ??ฑ??ผ??ล??อ???ำgฟ?ซพ@Vo?ฬA๑>jฎ@ฟล\ฟ฿๚=o?๕/M?1ฉ๋ปข+Eฟ{*~ฟ$ฟUH?;:?ot?w?J์3?tํ>q}*พsิฟUฟzฟ?๗}ฟgฟฌ็<ฟ๊บฟ~&พG๒pฝ๛&>า3น>g?(๋,?;ีI?jป_?*Lo??Qy?์~?๛?ๅ$~?รy??is?k?wฐb?Y?ิO??นD?ืe:?+0?!$&?.e????K๕	?ํT?ฤ>๒>ฌโ>๑ำ>N
ฦ>๒ธ>?ฃฌ>|ก>?=>
>ผ>Qhs>อb>fJS>ผฯD>๛L7>ฟฒ*>^๒>ๆ?>&ศ	>งD >Lฯ๎=-L?=*๋ฮ=ลภ=ฝBณ=๘ืฆ=yH=M=~=Zz=]?h=SิX=ปษI=ูษ;=Hย.=ไก"=ณX=ีื=p=6๑๓<คใ<ส@ำ<ัฤ<U๑ถ<F>ช<ึl<`m<R1<@V<Um<j]<กรM<คz?</2<ฃะ%<M<)<<rฐ๘;nl็;F[ื;ฌgศ;อ}บ;8ญ;ม~ก;nH;bู;ศ#;~5r;da;พQ;ฎ.C;ก5;a);?-ู>ลqฟณตพNฬa?k(?Nฟt~ฟ๊5ทพ
?N??BG#?ฮต๔ฝฦwDฟ?ฟ?$Jฟ%พฆ?>์6?ฦau?Gm|?sT???[>Mษ พ?+?พ0ฦ,ฟญYฟพtฟฟฌ|ฟBชnฟSXฟึล<ฟ}zฟAึ๘พฆ่ตพูfhพ0ิฝ{2I<(๖=ฑค`>>&\ศ>๎ใํ>๛ด?บ?ั#?แy/?cพ9?จรB?hฌJ?ฅQ?ฅW?ฃํ\?a?e?s	i??l?ๆถn?(q?) s?่บt?";v?zw?ฉx?Tคy?ถ}z?&:{?z?{?k|?ฏๅ|?๗O}?ฌ}?า๛}?ํ@~?ห|~?ฆฐ~??~?x?*&?YC?\?r?y?ไ?ค?lฐ?ป?Sฤ?Rฬ??ำ??ู?p??๐โ?ีๆ?5๊? ํ?จ๏?ู๑?พ๓?c๕?ฯ๖?
๘?๙?๚?ี๚?๛? ??ฅ????|??า????^????ฦ??๐????4??P??g??|??????ช??ถ??ภ??ศ??8พdพtฟ฿บซ>?mv?]๑ะฝ3p|ฟ๊จ ฟXช?~5?หซ?>ฐรพ็qฟH)fฟเภพ>ุM?dจ?#a?ฐ?:ณ=ฤุฑพๅ1ฟiฟขuฟ2๓vฟ๗ศVฟ:&ฟ
ฺพBCพ6ธ =[๘>ฯ	แ>x5?S:?"T?3g?(Dt?้$|?)ฅ?ฺ?|?1:w?{.p?Hุg?=^?ฏมT?ยJ?@@?๘๓5?ผฬ+?ฮโ!?ึG?_?ื,?หt๛>7g๋>2?>sำอ>Gภ>๎ณ>ใง>้M>tฤ>ดๆ>wW}>Vl>๕[>)็L>ฒู>>eฝ1>.%>ฒ>S|>0>Bถ๘=g็=Gื=ฌศ=ฅตบ=pวญ=rฝก=!=?=rb=ปฏr=฿ฺa=H0R=oC=,	6=ชg)=Gฆ=ต=?=q?<mw์<?<Wวฬ<`พ<ํUฑ<{ฅ<ภ<่<ฮ?<ฉw<าSf<SVV<?tG<9<Dน,<tป <ภ<^0<<่๑;`Tเ;Kมะ;Cย;Uฦด;Z9จ;p;ญ;๎;
M|;ฦศj;ื{Z;PK; 3=;D0;xฟ}?J'พฅ+qฟฏ>	ช~?%8*>๛Q]ฟด4SฟJเ?=บง^?Ql?pง>ฌ(เพ1mฟใ่tฟช2ฟWธSฝร0๔>สX?๕?Ep?2o8?Mpั>N=ย๎พฬKฟฯoBฟgฟI{ฟอฟฤพwฟT๒eฟฟMฟQ/ฟmMฟุ?พa;พy1พ$Wฝมึu=#(>ฤโ>็/ฑ>กู>ึ??> `?d?ด
)?ฑ4?ฅส=?ดPF?oศM?ขPT?MZ?ฎ?^?fVc?g?@gj?Cm?+ฟo?ฑ็q?rวs??gu?2ัv?ฅ
x?}y?5z?าz?ฒ{?8|?Cข|?}?ny}?๙ฯ}?๑~?ใ[~?%~?เฤ~?๏~?ฅ?O3?ปN?{f?{???K?จฉ?:ต??ฟ?ํว?qฯ?๓ี???wเ?ฑไ?Z่?๋?D๎?ฅ๐?ณ๒?|๔?๖?]๗?๘?๙?d๚?%๛?ห๛?\??ู??E??ฃ??๔??:??w??ฌ??ู????#??A??Z??p??????ฃ??ฏ??บ??ร??7C?iNฟr ฟฏญ\?u?z&ฟMlฟ๎Juปกb?าฒ\?Hิม=?ฯ4ฟ฿ฟ<ฒ3ฟปฝ๙N?ฐ?n?ผฟz?>?ๅซ>_Q๐ฝ๗?ฟWOฟฟฆwฟk?~ฟ๘หjฟ`Bฟh3ฟ`ยฃพ#ญฝ3
>0jฎ>ๆ๗?5)?ฒ๑F?ษ]?๙ึm?ฒqx?a;~????$}~??]z?5t?>l?ปc?ฌ-Z?!'P?gโE?E;?CO1?ZA'?|y???๔
?ึG?]๔>Qaไ>*ี>าว>.aบ>ช๛ญ>ชWข>ไl>ง2>?>hXu>d>-๛T>ฉcF>ำล8>4,>: >V/>สไ
>?M>iฝ๐=hเ=ษะ=้'ย=dถด= 2จ=?=4ฑ=ซ=ภa|=#แj=Z=ีlK=๋O==R-0=ฤ๓#=&={?=ษ!=!์๕<dฺไ<ฯ๗ิ<`/ฦ<mธ<?ซ<ถ<ส<xN<{ด<5o<้^<QoO<คA<ใก3<K)'<V<?ม<ฺด<`ต๚;xM้;๊ู;<ส;rผ;๓๓ฎ;qฮข;ะ;?;K2;๔,t;9c;rS;eฤD;'7;%?m?0ฉ]ฟ๏มฟ}'L?'ะB?แ๑ฤพ?ฟ๐๎พ2น?ำู~?๛95?5? ฝ5U6ฟฆ`ฟอTฟ?)ธพVN>^+?$q?>;~?าZ?E'?bฐ>Bภตฝ<ฬพ๖&ฟคUฟrrฟAฟv}ฟ6ฐpฟ][ฟ@@ฟฬ!ฟ- ฟูfฝพ8๓vพ H๐ฝ<ฤน2?=ฎU>>8แร>เ้>'๋?Z?i"?ๆ<.?ง8?๏ฮA?ึI??P?ษW?ผ^\?a?e?ซh?๎ภk?on?5ฤp?_สr?8t?v?Uew?!x?็y?ฬfz?J&{?Dฬ{?\|?ยุ|?ฤD}?Xข}?j๓}?ฅ9~?|v~?/ซ~?ีุ~?^ ?"?F@?๖Y?5p?y?)?ข? ฏ?๗น?Zร?zห?า?ุ?ไ??wโ?lๆ?ฺ้?า์?d๏?๑?๓?7๕?ฉ๖?้๗??๘?๏๙?ฟ๚?t๛???????q??ษ????W????ม??์????1??M??e??z??????ฉ??ด??ฟ??ีt?ว๏ปา{ฟึุl>`+|??3ผNภwฟ>ฟ-A ????Pซ?ฒpฆพมlฟ๗รkฟบืพส๕|>ฮbF?๎~?v\e?ญ๕?พ3๔=?|ฃพ],ฟฝำfฟั?~ฟธdxฟRกYฟร?*ฟบอโพfTพข๚ม<ๆs>ัฺ>ล?ซC8?UR?f?โs?>ฟ{?ไ?สฅ?ิะ|?|จw?แทp?Luh?ชC_?rtU?ฦIK?๘@?{ช6?ห,?๖"?
๐?ๅฉ?7ว?ณ?>~์>้9?>yฬฮ>ข1ม>cด>[จ>)>nz>>	~>?m>ง]>ล๊M>๔ห?>_2>็U&>$แ>j3>ห?>'๔๙=ญ่=ุ=ษ=ะคป= ฆฎ=ฦข=%I=ชฬ=ด	=็s=ฒ?b=>S=D=?๒6=&A*=ฐp=่q=M6	=ฟ`?<งํ<9(?<Uฮอ<ฟ<ฐ9ฒ<pฺฅ<?V<?<ง<ภx<ซ{g<ฃiW<0uH<:<!-<๊!<แR<)ใ<๋,<F๒;tแ;sอั;<ร;ฎต;rฉ;T;:h;>;"};^๖k;~[;สUL;	&>;8พA??ผ=8พไyฟๆz0>ึ??์>๖Rฟ*]ฟ<cใU?r?ฤย>vวพภ*hฟเxฟฬ!ฟณใบฝ]jใ>ฐS?p,~?+r?G=?;g?> ๖ต=ถรwพลฮฟ/?ฟ9eฟฐnzฟ?ํฟ?งxฟFqgฟ?Nฟท1ฟ/ฟษเพาพ๐เ9พBๆxฝึV=ิ!>ฃ>ู?ฎ>๑wึ>	๚>qS?i?8(?[3?๖'=?
ยE?KM?wใS? ฆY?ฌ^? c??f?n0j?um?รo?บรq?9จs?ลLu?ฑนv?A๖w?ฯy?เ๖y?Eลz?-x{?<|?|?}?๐r}?Xส}?~?ชW~?}~?ตม~?W์~?D?@1?๓L?๐d?ทy?ต?K?ษจ?yด?พ?\ว?๔ฮ?ี?8??&เ?kไ?่?Q๋?๎?}๐?๒?^๔?ํ๕?G๗?r๘?u๙?V๚?๛?ม๛?R??ั??>????๎??5??s??จ??ึ?????!?????Y??o??????ข??ฎ??น??*ด>U.L? >Fฟrซ
ฟ'V?๕ไ"?๘Wฟfบoฟถ๒ฝศฤ^?ไg`?ช]๙=a'0ฟ??ฟฉ7ฟz}ธฝl;
?2m?ฐ{?Iฌ@?rรฒ>ฟGีฝ้%ฟjMฟk๑vฟ4ฟwปkฟฆฮCฟ้ฟ฿`งพsผปฝz>ซ>หผ?4(?)F?]?pm?m3x?๛~???7~?z??jt?ฑลl?งd?ๆxZ?uP?u1F?6?;?1?i'?4ร?ฺM?8?ถ?ฌ๔>%ึไ>?ี>ํ๗ว>)รบ>ผWฎ>ฎข>แฝ>}~>็>?u>ue>๋nU>ฐฯF>*9>4p,>ฑ >>์0>ส>A๑=เ=o
ั=ญย=ฮต=จ=ฬเ=m=Xโ=ว์|=bk=[=๑?K=Bธ==o0=%N$=C็=รJ=คj=บs๖<Xๅ<?mี<ชฦ<?ำธ<ฟ?ซ</?<ม๑<ฟ<v๛<Qp<๒c_<นแO<sA<๕4<}'<!ไ<w<&?<ง?๛;&ฮ้;ฉู;ชwส;$iผ;rTฏ;<(ฃ;`ิ;ืI;จz;กณt;nถc;.็S;์0E;Wmvฟl?ต๙!??0Wฟiฟ฿|E?จ'I?g?ณพฒัฟIE?พ๓e๖>a~?่ม9?p ผศV2ฟ๕~ฟงxWฟวภพใฦ=>ู(?ญใo?ง~?วr\?K?ล๛>?kกฝกฉวพ"้$ฟธeTฟ"๏qฟ์~ฟ่ฟ}ฟ|5qฟm\ฟc?@ฟ]"ฟึ,ฟ๔eฟพ8ีzพธญ๗ฝุ_vป~ศื=ฮพR>ไB>ฆญย>อ่>Jp?D?	"?ว็-?ญ\8?2A?jI?ผชP?ลีV?U8\?๊`?๐e?ทh?฿ชk?i\n?ณp?้ปr?ซt?ปv?ใ[w?๏x?ฬy?ค`z?๓ {?ฃว{?X|?Iี|?มA}?ผ}?(๑}?ฐ7~?สt~?ทฉ~?ื~?D?~?ฉ!?r??>Y?o?๐?ฒ?5ข?วฎ?ชน?ร?@ห?Rา?qุ?ฟ??Vโ?Pๆ?ม้?ผ์?Q๏?๑?}๓?+๕?๖?เ๗?๗๘?่๙?น๚?n๛?????	??n??ฦ????U????ภ??๋????0??L??d??y??????จ??ด??้ใ)ฟ+ะu?C?กผ7E{ฟiw>ถ{?eBผMxฟ็ฟ๙๋?๘?๖D?2aฉพMmฟฝ<kฟ6sีพ`ฝ>G?ง?*๑d?C<?๛ปํ=B์คพไโ,ฟ,gฟ
ฟ_Axฟ๎YYฟ?แ)ฟ|๛แพฐปRพnฉฮ<ว?t>gp?>฿แ?xx8?ธR?n0f?รs?ษ{??.ฃ?ษ|?ขw?Pชp?ฤeh?ศ2_?ธbU?7K?Kๆ@?[6?n,?จ"?S฿?ื??ท?{}?>Wb์>ญ?>ตณฮ>Nม>Mด>kFจ>ื?>Th>>Yv~>s!m>?๏\>๒ะM>ฺณ?>ๅ2>๑@&>อ>3!>ำ.>ิ๙=่=ฃzุ=๗ษ=ป=๙ฎ=%xข=๒5=สบ=๙="ศs=?฿b=5#S=}D=?6=+*=\=*_=?$	=I@?<โํ<?<,ดอ<ฦlฟ<#ฒ<Zลฅ<`C<?<?<๓?x<=^g<?NW<ณ[H<ศr:<-<`u!<ฤ?<`ั<^<ย'๒;?Wแ;ฦฒั;ผ#ร;rต;๒๛จ;@;U;พ,;ไp};]ุk;x[;ฮ;L;ณ?ฟน?พ/๓?(ํCพ[gxฟl:>๚๕?35y>KSฟ๑\ฟw<eอV?๎q?วภ>?ษพDฑhฟoฦwฟ?๔ ฟๅึฒฝ=ๅ>ฃ-T?๕D~?๗Zr?gอ<?+8?>|ฑ=๘yพมAฟบ?ฟทeฟ9zฟ%๋ฟทxฟจKgฟuฯNฟA1ฟ่^ฟJเพh]พk 9พuฝ0์Y=ท!>็๕>ฝฎ>>ปึ>/ฟ๚>3n? ฐ?tM(? n3?-8=?BะE??WM?Y๎S?ฏY??ด^?8c?โไf?ๅ5j?5m?ไo?Pวq?Vซs?xOu?ผv?J๘w?
y?g๘y?ฦz?Sy{?;|?y|?ฬ}?s}?่ส}?~?X~?ฺ~?ย~?์~??t1? M?e?ูy?ำ?d?เจ?ด?ฉพ?kว? ฯ?ี?A??.เ?rไ?#่?V๋?๎?๐?๒?a๔?๐๕?I๗?t๘?w๙?W๚?๛?ย๛?S??ั???????๏??6??s??จ??ื?????!?????Y??o??????ข??ฏ??ค๏|ฟFณ>ฒฎ??|WQฟV๘พู^?^6?'๎(ฟ๏jฟI<d?:A[?mๆฌ=ธ6ฟจสฟน.2ฟง~ฝิ?eyo?lz?p=?Nฉ>!z๚ฝแฟ๊?Oฟ้wฟ*็~ฟpjฟ๙ีAฟ/ฟFeขพ8จฝร>ฏ>&n?j)??<G?kั]?4?m??x?ืE~?๙??ฌt~?!Nz?c!t?Fnl?z?c?OZ?ย	P?ฤE?p;?๑11?ธ$'?ฝ]?$์?oฺ
?m/?๗?๓>\5ไ>?dี>งhว>Q<บ>ูญ>)7ข>mN> >_>&u>โmd>คฯT>;F>๊8>ุ๎+> >>'ศ
>,3>ฒ๐=๊฿=ฉlะ=ม?ม=?ด=/จ=6j=={=u-|=vฐj=MiZ=ชBK=ฌ(==ห0=ลั#=s=	฿=b= น๕<ํชไ<ฃหิ<Eฦ<IGธ<|ซ<๚<๖<ว1<ศ<Xo<รบ^<IDO<เ@<|3<'<n<ฃ<่<]๚;้;เํุ;S?ษ;qฺป;จฯฎ;ชฌข;ba;ี?;;L๚s;๗	c;ฐFS;๔>๗ืoฟฐ)?Y?๏๕_ฟ๏๛๛พตN??Y@?dหพZ?ฟิน่พ &??่}3?;%ฝ๙ฯ7ฟ?ฟิฦSฟ๏๙ดพT>b,?Kq?~??3Z??A?%c>?dฝฝฉอพ29'ฟ๖Uฟศrฟ"ฟ&p}ฟ}pฟม[ฟ9ฦ?ฟอฟ ฟo?พaฆผพ}uพจํฝดฉk:เ=6ษV>c>ำTฤ>2H๊>R?W/?ำ"?โ\.?ณร8?ข็A?ฆ๋I?๐P?QW?)m\?a?	*e?ดh?7ษk?ศvn?vสp?อฯr?๏t?ถv?โhw?4x?y?iz?K({?ฮ{?]|?ฺ|?ๅE}?Sฃ}?C๔}?b:~?w~?ผซ~?Oู~?ศ ?๙"?@?;Z?qp?ญ?V?รข?Bฏ?บ?sร?ห?า?ญุ?๒??โ?wๆ?ใ้?ฺ์?k๏?ค๑?๓?;๕?ญ๖?ํ๗?๙?๒๙?ม๚?u๛???????r??ส????W????ย??์????1??M??e??z??????ฉ???ฮพฏเ
ฟUr}?ฃญพีbsฟณ>mXu?งศํฝ4๚|ฟน?๛พ1?๛~?A๘>ฎศพํrฟิBeฟ2ฝพJq>N?vป?zS`?{{?MCฉ=๔?ณพVN2ฟฆjฟฟัธvฟ6YVฟ<๕%ฟ๎BูพึAพ|P*=>ต๘แ>?wข:??]T?ั]g?แ_t?ห3|?Xฉ?R?{|?X)w?p?xภg?o^?ฅฆT?wJ?พ$@?lุ5?ธฑ+?ศ!?x.?๐??~H๛>0=๋>S
?>ํญอ>ถ#ภ>ณeณ>ฏlง>ฅ0>ฉ>อ>z'}>}่k>ฎห[>ภL>5ต>>\1>qc%>?>ภ`>~{>b๘=ฮX็=ฝXื=?qศ=?บ=ๆฅญ=8ก=k=์?=BI=ึr=9ฏa=จR=uC=๛ๅ5=้F)=ฬ=)=l=M้?<ฑI์<ใ?<ปฬ<kพ<3ฑ<ๆค<t<์ฬ<ใ<ศRw<D'f<?,V<gNG<-x9<ฺ,<\ <ัu<q<{m<Eโ๐;๚(เ;้ะ;pย;\ฃด;ฯจ;'m;๋;ตu;;|;[j;QZ;,j?IWฟxDพ}?;ผพษปoฟ?,>ฦD~?ไ>ษแ^ฟcQฟถ=ฎไ_?Qงk?fฃ>ชัใพๆmฟยctฟฦเฟYG;ฝDฐ๖>๘5Y?ส?ซจo?Nฑ7?ใฯ>D{=Wพ๗ฟ๋Bฟฯgฟrh{ฟRวฟwฟปทeฟอLฟd9/ฟ๔ฟล#?พ)พบ/พwRฝฉz=น5)>ไ_>กฑ><ู>9?>^?)ช?K*)?04?ใ=?#fF?2?M?aT?Z?*_?Fac?'g?|oj?FJm?cลo?ํq?"ฬs?๏ku?บิv?ตx?%y?z?ิz?m{?ท|?ฃ|?ฌ}?hz}?ัะ}?ญ~?\~?ฒ~?Zล~?๏~? ?3? O?ทf?A{?
?r?ษฉ?Wต?Xฟ?ศ?ฯ?ึ?ค??เ?ผไ?c่?๋?K๎?ช๐?น๒?๔?๖?`๗?๘?๙?f๚?'๛?อ๛?]??ฺ??F??ฃ??๕??;??w??ฌ??ฺ????#??A??Z??q??????ฃ??M*?บmฟผ?บศ?ภ1hฟexชพ+o?/๐>ฬAฟw\ฟฟV?=j<o?BๅL?ซฺผypEฟd~ฟฉื#ฟ๗็<+T?์t?=w?ำฝ3?๘x>j\+พ ฟตUฟ?zฟG๒}ฟgฟใฮ<ฟ?ฟฒ้พบoฝฃ๑&>cน>?๛,?ํแI?ภฤ_?Ro?กUy?~?แ๚?H#~?ฬภy?Cfs?\k?หซb?Y?ต?N?ดD?ซ`:?&0?)&?^`?=๘??๐	?ดP?ฝ6๒>tคโ>ึ้ำ>ฦ>?์ธ>Bฌ>?ก>บ8>>>ฒ_s>๚ฤb>แBS>ธศD>oF7>คฌ*>ฌ์>๘>5ร	>@ >ทฦ๎=.D?=ทใฮ=ึภ=H<ณ=๕ัฆ=แB==ฅ{=?Pz=๕๔h=ฬX=sยI=ร;=๚ป.="=<S=ภา=ต=g่๓<r๚โ<)9ำ<ธฤ<บ๊ถ<!8ช<g<h<^,<M<มm<n]<3ผM<บs?<)2<ฆส%<H<๚<H<xง๘;d็;Sื;o`ศ;wบ;๓ญ;์xก;C;Vิ;;ฟ,r;t\a;ชU?lมฝclXฟย$K?ฺื>ะdqฟLดพU b?(?บฟฅ ~ฟo;ถพ๘x?^???๓"?Dิ๗ฝดณDฟเ?ฟ๓Iฟพc>ษI6?su?ืc|?5VT? ๑?ุZ>qพ8s?พEแ,ฟฟYฟิวtฟ?ฟ0|ฟ??nฟXฟูถ<ฟ6jฟฌด๘พ9วตพ&hพ ิฝxฬL<๖=aี`>2ง>pศ>ผ๕ํ>๊ผ?ลฃ?;ื#?^/?6ร9?ๅวB?ฐJ?ไQ?jจW?๐\?ตa?je?i?hl?"ธn?:q?s?ทปt?ี;v?w?"ชx?ษคy?~z?~:{?ว?{?Ik|?่ๅ|?(P}?7ฌ}?๗๛}?A~?็|~?พฐ~?ฅ?~??9&?fC?ซ\?r??์?#ค?rฐ?ป?Wฤ?Vฬ?Bำ?Aู?s??๒โ?ืๆ?6๊?"ํ?ฉ๏?ฺ๑?ฟ๓?d๕?ะ๖?๘?๙?๚?ี๚?๛? ??ฅ????|??า????^????ฦ??๐????4??P??g??|??????9๚?ฝK&ฟ๒พn~?๓?ภพok\ฟR?F=c?!พฬ?ฟANตพฦ2/?vรy?ฟ>พt?พ้yฟฑใXฟพซดม>WY?{ๅ?HW?/๒>Zผ<^ฮพ;ฟ๕ปnฟ๗ฟฏจsฟ,มPฟ5ฯฟ9ษพ!พ?ฎ=สj>{ํ>ง[?Jj>?2:W?|ei?ญญu?@ใ|?ี?ํE?ๆ{??Rv?ึo?&f??;]?nTS?I?๐ษ>?๎4?a*?d ?น๒??ภ?๔?!๙>ฌ2้>ฺ>?ห>{lพ>ปศฑ>๘่ฅ>ล>}T>V>gำz>sปi>kรY>cฺJ>๐<>ฅ๔/>Fู#>ฉ>J
>d<>ฯ3๖=๗.ๅ=้Tี=ฦ=Rาธ=kฌ=x?==โญ=t=:p=1_=)P=?A=๖04=.ฐ'=B=฿8=บ$=๛<ษ๊<าู<ๅณส<ฮกผ<ญฏ<-Zฃ<+<ฆu<ขฃ<; u<??c<?)T<EoE<Nบ7<๏๘*<?<<ว	<U6 <+๎;?;mฃฮ;ลJภ;๑ฒ;ฐฆ;๕;๗2;0;"พy;Vgh;ทYผุกB?`tฟฺWโ=I$m?Y2ฟฏXฟ*ศ๋>Z$u? "ผsioฟญจ:ฟฆ`>hm?Vท^?)๛]>
?ฟิuฟธ๘lฟ็๛ฟ้<V
?ๆua?บ๎?ญHj?B8.?|ุธ><NพD.ฟษHฟGTkฟyห|ฟA\ฟmภuฟ๏หbฟย$Iฟบ+ฟ
ฟgZาพ๋พL0พ:ฝ"8=ฃ6>ึn>iท>k?>ถฮ ??{?ธd?ฑ*?5?^??)oG?(รN?ย+U?ฤZ?tฆ_?ญ็c?g?=ีj?นขm?=p?ี/r?t?6u?Y w?3x?๔=y?๔$z?6ํz?ฬ{?=1|?ณ|?$}?u}?A?}?ธ$~?[d~?{~?;ห~?๔~?i?o7?NR?i?ผ}?0?N?fซ?ผถ?ภ?ษ?kะ?ฬึ?R??แ?>ๅ?ิ่?๏๋??๎?๔๐?๘๒?ท๔?:๖?๗?ฌ๘?จ๙?๚?>๛?แ๛?n??้??S??ฏ?????C????ฒ??฿????'??D??^??s??????๗q?w?b>ฐkฟD?.ถ>}|ฟjฃณฝบ๚}?;>ฟ?^ฟท%@ฟ่>๑?z?I3?{พกYฟymxฟffฟ7F้=.x0?rz?Kbp?Bฏ$?งY>%upพธOฟ?ป]ฟtผ|ฟฏ๏{ฟ}aฟใ4ฟz฿๚พIึพ=ทผฑฝH>2ศ>ํื?_2?aวM?Ub?โ>q?ปuz???fโ?ฐ}?ฃ?x?ถ@r?X:j?B2a?N}W?`M?}C?เม8?ร.?$?฿?ฃ?*?N??>กด๏>oDเ>ซั>8ๆร>๓๎ถ>าพช>N>>P>ภ!>ฏp>A`>*๊P>B>j;5>แฤ(>์%>ปP>78>??=์=ล?=๗ฬ=hพ=8ฑ=ี๑ค=ใ=๚฿=J๘=?w=ปUf=ฝ[V=๕|G=?ฅ9=Fฤ,=;ว =๓=ท<=ี=)๑<๊kเ<
ุะ<?Xย<5?ด<7Mจ<F<๋พ<พ?<ฒl|<ๆj<ฦZ<ฮjK<K=<7'0<t์#<	<฿๓<ํ<Cฺ๕;ฉศไ;kๆิ;yฦ;;]ธ;uซ;,ง;;A;รง;8ro;็;Yฟฃกy?|ศพ-๕#ฟ?Ao?r{'>hฟMg พีธv?x๛>0))ฟถtฟ๔Kพ-ผ6?ฏ}?D?'7wพงVฟ{U~ฟ(u9ฟV&Tพ๗ฐ>U?C?P(z?ืx?๕ผJ?ๅ?>w?">๙บ5พตg๒พ?&5ฟต)_ฟwฟก๏ฟb{ฟokฟ=qTฟ๘7ฟBIฟ?(๎พฆNซพแSพงญฝy2๖<ัเ>?o>Jฅ>ฮ>๓>,4
?ฬำ?]ฦ%?ุ31?C;?ญD?ืK?ุR?ฑX?ด]?๔5b?"f?ฯi?4l?'o?_Xq?Ks?เ๛t?sv?`นw?ิx?ษy?z?ศU{?m๕{?ว|?ซ๗|?_}?น}?~?K~?~??ธ~?&ไ~?+
?+??G?T`?นu?@?L?1ฆ?:ฒ?งผ?ญล?~อ?Cิ? ฺ?3฿?ใ?g็?ณ๊?ํ?๐?+๒?๔?ก๕?๗?9๘?D๙?+๚?๓๚??๛?6??ธ??)????฿??(??g????อ??๖????9??T??k??????ย:?พj?๕๙dฟ๓=ฤณe?z&ฟ?,ฟะย>?ฉR=?sQฟI๘xฟวขพP?k?1Z>|ฟSTฟ๓sDฟHิ"พg
๘>)๚f?~ำ}?ำ0I?้ฮษ>*_rฝ๑พ฿{GฟStฟxฝฟ&ยnฟ?Hฟๆฅฟ?ณพี๋ฝXIโ=Mก>Q?>ย$?~vC?d[?ูl?ฌXw?ธ}?_๘?_ู~?.{?ไu?Mm?เ๎d?iv[?|Q?<G?9่<?ฏค2?(??ผ?>?ฉ?พd??)๖>bๆ>=sื>DYษ>ยผ>Mฏ>hำฃ>ฺะ>?>๐ื>w>`ฝf>?๗V>>H>๊:>vฏ->]ป!>>3>บ>{๓=ฆ5โ=?า=R?ร=zkถ=๔ศฉ==๊= เ=ล~=m=1จ\=พYM=ฏ?=Lุ1=%%=๛=ซT=b=T@๘<<็<(?ึ<ๅศ<ท,บ<>@ญ<^9ก<-<ื<์<ฯq<a<VfQ<ฝ?B<~U5<ฉพ(<<ช <๛<Z?;=๋;h)?;-๒ห;_ษฝ;9ฐ;BYค;8๐;๚Q;tp;}v;_Xgฟ๊:ฯ>๖ไ>(`ฟัโ>พ\B?q=ฟน*ฟํP,?Y\?๏Snพ?}ฟ4ฟีศ>?z??<H?$=$$ฟ'พ|ฟ๚๓_ฟธฦ?พ*>QL?kFk?*?ขทa?q ?Lู>8ฝ~ธธพlฟlPฟ?ปoฟ	L~ฟบn~ฟr้rฟ	^ฟ๏CฟQ%ฟฟ'ฦพ ?พภcพ๛๙yผจ๚ม=หมH>ชน>พ>d"ๅ>นอ?%?ภ ?ล,?V]7?ญ@?๙ืH?แ?O?ข?V?Tต[?Xx`?}d?-;h?_k? n?ปzp?r?ึTt?โu?ฅ;w?๙fx?jy?Kz?ป{?ูท{?jJ|?mษ|?{7}?ี}?q้}?1~? o~?ดค~?8ำ~?๛~?g??<?อV?ym???ึ??ญ?ขธ?2ย?{ส?งั??ื?>??็แ?๏ๅ?n้?t์?๏?X๑?O๓?๕?{๖?ย๗??๘?า๙?ฆ๚?]๛??๛??????e??พ????O????ป??็????-??I??b??w????8ๅ}ฟ์๏^?tพv?+ฟu?1ผBซ{ฟB็o>?
|?hพ0ผํ่wฟ'ยฟบ ?ซ??ดE?Fงพ	้lฟผkฟฃืพ}>~>งF?8๕~?>e?ม?]๒=;ๅฃพ,ฟY่fฟฟ ฟถZxฟYฟ%*ฟ
โพUํSพฐล<๗s>ฉ?ฺ>ฒ?ซR8?ลR?ฤf?@s?.ย{?ฮ?ฅ?ฦฮ|?hฅw?ดp?ไph?฿>_?joU?DK?`๓@?Vฅ6?พz,?"?K๋?Vฅ??ย?g?>1v์>v2?>qลฮ>+ม>J]ด>+Uจ>ฌ
>Ju>6>
~>จ6m>ฬ]>pใM>ล?>?2>๓O&>?>>.>๚:>,๋๙='ฅ่=Iุ=Cษ=ป=ีฎ=๊ข=ฑC=ว=๛=P?s=๔b=l6S=jD=A์6=;*=๘j=l=Y1	=W?<ํ<> ?<็ฦอ<5~ฟ<@3ฒ<sิฅ<lQ<๏<หข<ทx<Osg<?aW<๓mH<ร:<?-<!<tM<?<8(<ะ=๒;dlแ;เลั;5ร;?งต;Wฉ;ุN;๑b;&9;๚};๊พจฌ๛พฒx?Pฝ=ฟตพ&??๋;พเxฟ.\3>0??me>๛sRฟทG]ฟฺ5%<&V?ฏ๓q?Zย>ฯ5ศพQhฟ?๛wฟm!ฟธฝฮไใ>ภำS?x3~?r?$=?4?>Rด= dxพp๏ฟึF?ฟ๖eฟuzฟ๐์ฟ??xฟfgฟ๓๑Nฟง1ฟฃฟ$`เพ๚ฐพ.ก9พa๕wฝฦถW=l5!>บ>Uฎ>ึ>๚>
[?%?u>(?ส`3?,=?ฦE?OM?ๆS?ณจY?๎ฎ^?c?cเf??1j?ฯm?๏o?ฟฤq?ฉs?Mu?[บv?ี๖w?O	y?O๗y?ฅลz?x{?|??|?C}?s}?ส}?5~?ษW~?~?ฬม~?k์~?U?O1? M?๛d?มy?พ?R?ะจ?ด?พ?aว?๗ฮ?ี?:??(เ?mไ?่?R๋?๎?~๐?๒?_๔?๎๕?H๗?s๘?v๙?V๚?๛?ม๛?S??ั??>????๏??6??s??จ??ึ?????!?????Y??o????๊พ$ฟห
>ิป?ไฟ?ห["?Uฬdฟโธพ0#m?q๛>บ6=ฟJ/_ฟ	ื=	m?O?zaๅ;7?Bฟฮ~ฟ๔[&ฟQsปย?mนs?;ฉw?โ^5?y>#พ๊uฟ^มTฟ฿ผyฟx!~ฟLฆgฟฐญ=ฟWฃฟ3พปฝ?#>ไดท>ฐๆ?๎g,?RoI?tp_?!o?z3y?๎~?`??ท1~?cูy?s?gทk?ุีb?ย9Y?ี*O?มโD?8:??S0?่K&?ณ?๑!?ส
?ลv?๒>๖่โ>*ิ>@ฦ>จ%น>3ำฌ>zAก>)h>>>ซป>_ญs>c>ฃS>๕E>n7>ชใ*>๙>i(>ฤ๏	>i >๏=:?=ั&ฯ=Rะภ=vvณ=!ง=Ou=
ฏ=Yง=Yขz=ฐ@i=?Y=J=# <=ฮ๔.=้ะ"=v= =W7=ย7๔<KDใ<ใ}ำ<ญฯฤ<?&ท<oช<ฉ<<Y<?<แm<`]]<&?M<ฒ?<c2< &<=z<ณภ<ยล<c๘๘;`ฏ็;ื;ฅกศ;ภณบ;lฝญ;xญก;็s;ื;nI;-๒C?๕}ฟปคO?sHํผแ]ฟณํE?;ฌๅ>เnฟYไภพฉ_?o,?ห๚พฝ~ฟeฟพj๏?e??6?%??ณ?ฝ๛Bฟ?ฟฐKฟDฅพf>ุช4?Rีt?%ท|?]U?y`?%a>$
๗ฝค๐ูพต์+ฟYฟYstฟ$ฟ,ป|ฟ3๔nฟ๔๒Xฟถ==ฟร?ฟใ๙พM๔ถพ๒mjพ๘tุฝ.X,<ำ๒=_>4เ>ฮผว>>Uํ>gu?Od?#?โM/?ฐ9?ฒกB?จJ??~Q??W?ะู\?Fwa?~|e?\?h?l?ญn?๙p?ณ๘r?nดt?5v?w?aฅx?ช?y?zz?e7{??{?๕h|?ไใ|?iN}?ณช}?ง๚}?๋?~?๋{~?ไฏ~?่?~?ๆ?ฌ%?์B?A\?1r?2?ง?็ฃ?>ฐ?๏บ?0ฤ?4ฬ?%ำ?(ู?]??฿โ?ฦๆ?(๊?ํ?๏?ั๑?ท๓?]๕?ส๖?๘?๙?๚?า๚?๛???ฃ????z??ั????]????ฦ??๐????4??O??g??{??ฝ>ึd8ฟ?-?T6ฟaพ์{?แพ๋ySฟะ2?	;\?5ญพ๗ฟข[พ๏6?\w?ชฎ>๏พฟ
{ฟูวTฟชพแมอ>ซ\?*ฌ?QbT?้ำ้>ฝยง;ึพื=>ฟฦ	pฟซ?ฟฉrฟ!Oฟธขฟๆาฤพฒพฝlข=ศb>๖โ๐>$ร???MX?n?i?v?@}?-฿?w0?ธ{?1v?มn?:f?gฺ\?๎R?สตH?ุa>?๏4?9?)?p ?<?Wf??ๆ|๘>่>mู>9Pห>J้ฝ>fMฑ>0uฅ>X>ะ๎>0/>u!z>+i>(Y>lIJ>รh<>{v/>กc#>?!>ค>(?>^๕=ฎไ=๙บิ=Dฦ=ีLธ="ซ=มฆ=\=I=ณ=บo=q๏^=xO=๒A=ฎ3=ฮ6'=M=พฯ=ไย=๖ะ๚<Ph้<ื4ู< !ส<8ผ<
ฏ<ใใข<<6<ND<ะNt<โXc<YS<KเD<A57<}*<ง<Iฃ<Mc	<๕ฒ?;\๒ํ;Dm?;วฮ;ฟฟ;oฒ;ฆ;?;ส;?ฮ;ตt?ฆ1ฟaำฃ=4x3?ษผyฟลW=>ณะe?EฟEPฟ&?๚๎p?หwpฝษHsฟ3ฟฝๅ>?มp?สHZ?_)>>ษVฟYzwฟkjฟลฟ4O=๎?\พc?$??่h?$M+?p๖ฑ>\P;:ฃพัฟ?|JฟTlฟ*)}ฟ1ฟ๗)uฟๅaฟฑHฟ<?)ฟ๊M	ฟgทฯพ6Vพผ>พKฝy?ข=ภ{:><>mปธ>มแ฿>ฤu?x?5่? %+?ว๎5?k??ืฝG?O?ํgU?๙Z?=ิ_?d?ศพg?m๓j?๗ผm?
)p?กCr?Bt? ญu?Iw?ร>x?ฏGy?d-z?๔z?#ก{?ป6|?_ธ|?ด(}?}?Z?}?f'~?ฎf~?~~?๙ฬ~?๖~?ท?8?IS?mj?x~?ำ???เซ?&ท?้ภ?^ษ?ฐะ?ื???Gแ?eๅ?๖่?์?น๎?
๑?๓?ศ๔?I๖?๗?ท๘?ฑ๙?๚?E๛?็๛?t??ํ??W??ฒ????F????ด??แ????)??E??^??t??Kปv?ต|ฟY๎4?ีEC<ถdSฟป0_?๋D>๘?ฟ๒t=๛?>5Xlฟ๗F,ฟ๋ษ>ค~?"J"?ส์lพ๐๑bฟeXsฟi๛พธ3>๎้:?ฑ๐|?Rชk?3[?ู->้?พ๋ำ#ฟbฟ๗ํ}ฟhqzฟ'๛]ฟq0ฟไ๏พปoพอ8บ2y\>^ฬะ>๐q?ํ๊4?IP?ผ;d?7Rr?{?M??ส??@}?Sx?คq?0li?ฆP`?SV?๎kL?ฝB?ฬ7?ฅ-?6ฆ#?๘๛?Oซ?Sฝ?ko?>ๆ:๎>เ??>ไYะ>จย>ซรต>นฅฉ>UF>>ก>lI>o>ฺว^>yO>๒NA>_4>ฆ'>๚>์W>]P>	๐๛=?๊=ธNฺ=โ3ห=๐"ฝ=์	ฐ=ุฃ=}=ภ๋=๒=ุu=วหd=ํT=ญ'F=:h8=ฆ+=ด=ไ=jN
=ต =M๏<ล๋?<rฯ<.ม<ฅณ<-ง<(<hล<ธ<ผz<bTi<!Y<J<r<<ก๙.<ฮำ"<฿<ึ <ร6<W5๔;๖@ใ;้yำ;Fหฤ;!ท;ธjช;ิ;9;NT;l>?-'>5ฟX๛?)dฟ|ฦ๚พk={?๘ท;ถฟฦx?<บj}??พฤ>น[=ฟสmkฟแีฝ/?E?ดy?ธ้์>0?พN๙^ฟฤ|ฟ'ํ.ฟ4ูพ๚ว>xK?H|?Q5v?ัถD?Y6๐>ก๘>;3TพZ?พณึ9ฟ?)bฟ}์xฟ๔?ฟ๖?yฟลพiฟฯ๙Qฟ๖5ฟ';ฟ๋่็พชฅพ๊๑GพัฝxG%=F>?x>ยฉ>ค<า>บ๖>ค?n?็&?;22?ฆ"<?ู?D?แL?4S?ุY?'^?กb?Kyf?Mุi?ืฦl?0So?่q?vs?4!u?ไv?xีw?`์x?9?y?ๆฏz?งe{?/|?ณ|??}?~h}?Lม}?;~?เP~?~?ผ~?๎็~?r?๐-?J?tb?w?ู?ฎ?dง?Dณ?ฝ?tฦ?+ฮ?ุิ?กฺ?ฃ฿?๚ใ?ป็??๊?อํ?=๐?Z๒?.๔?ฤ๕?#๗?S๘?[๙??๚?๛?ฏ๛?C??ร??3????ๆ??.??m??ฃ??ั??๚????<??V??m??ผฟ>?ท๊พ?ะฅฝ%ศ:?ื}ฟ|ๅด>/??ฮืQฟ%๖พ8_?่ค?ํk)ฟ้ฐjฟ๛7<ฯCd?@ [? Bฉ=๖ิ6ฟ~ฦฟ๋1ฟ"xฝ?3o?๊]z?ๆ<?{ฺจ>G>?ฝ9ฟงPฟ๔wฟ4ใ~ฟ`jฟฯฝAฟjqฟ(ขพuJงฝ๏>ฑฏ>ณ?'ฆ)?ํIG??]?ึn?แx?ฅG~?๕??0s~?~Kz??t?jl?ืc?_Z?ฆP?pฟE?^k;?ื,1?ผ'?้X?~็?๛ี
?-+?ๅี๓>ถ-ไ>f]ี>ืaว>่5บ> ำญ>1ข> I>*>ป>าu>ฦed>ศT>๕3F>R8>ฒ่+>Y >C>,ร
>.>๐=โ฿=)eะ=ล๘ม=ด=!	จ=d=ศ=ฑv=]$|=?งj=laZ=U;K=ุ!==q0=?ห#=n=ู๊==Aฐ๕<ฌขไ<๔ริ<?ล<ข@ธ<Nvซ<7<{<ส,<#<เOo<ธฒ^<อ<O<กู@<$v3< '<th<`<<Qx๚;ง้;ๆุ;ืษ;ฉำป;Xษฎ;สฆข;๊[;ฟู;tผ*ฟฦc?ื(ฟ9/?ทซ>B}oฟc*?ฅข?bZ`ฟนช๚พแ๛N?แ๊???ฬพใ๛ฟษ็พฒ?อ??03?+ฝ8ฟ=ฟ๐SฟาkดพขU>๎ว,?ญq?~?"Z?+?\ฑ~>ฬธพฝ?๑อพ0U'ฟl	Vฟ?าrฟ?$ฟ;l}ฟงtpฟS	[ฟkท?ฟฏ ฟๅw?พๅผพ <uพผํฝ*?:vแ=\๚V>ฑ.>ํhฤ>4Z๊>Y!?x6?!"?qb.?ศ8?ํ๋A?i๏I?W๓P?1W?ซo\?ฤa?๐+e?Dถh?งสk?xn?หp?ฟะr?ภt?lv?iw?ฝx?	y?iz?ฅ({?Nฮ{?แ]|?Kฺ|?F}?~ฃ}?i๔}?:~?;w~?ีซ~?dู~?? ?	#?ฃ@?GZ?{p?ถ?^?สข?Hฏ?บ?wร?ห?า?ฐุ?๕??โ?xๆ?ไ้??์?l๏?ฅ๑?๓?<๕?ญ๖?ํ๗?๙?๒๙?ย๚?v๛???????r??ส????X????ย??์????1??M??e??๛m"พดณเ>สOฟฅค?ฟK)ฉพ+4?บฒพEํ_ฟัถ??e?๛พ.?ฟ}พพฆำ+?<ชz?3ๅฦ> ๖พ%yฟZฟฆAพiผ>m&X?๔?ฤ{X?.+๖>jํ๙< หพฯm:ฟอ*nฟิ๏ฟษtฟzQฟlนฟหพ	%พิ*=?ป>๊์>ีย?F๑=?ํ?V??$i?u?	ฮ|?cะ?ฉN?๔๙{?nv?3o?Wปf?+e]?kS?{II?๎๕>?yฌ4?ผ*?าช ?ฒ?W็?น?Fg๙>รt้>Zฺ>ฬ>คพ>๑?ฑ>๛ฆ>๓>>ถ>ฝ{>ฺj>+Z>ฤK>K)=>*0>$>พ>5>ทd>๑~๖=๓tๅ=ี=Rฮฦ=ื
น=:ฌ=wK?=ฒ/=Yุ=๚7=p=ฏี_=่NP=f?A=/h4=ใ'===be=(N=+ิ๛<Y๊<Vฺ<
๒ส<ค?ผ<ฟฏ<Cฃ<ว1<ก<?ห<[Ku<ๅCd<kT<ะซE<ฅ๒7<\-+<K<่;<T๑	<ฅ] <X่๎;,R?;หโฮ;ฝภ;๗'ณ;ภทฆ;$;/_;Dย|ฟaf?๙ฟ@2XฝหH?นqฟfก=Rๅo?"๘พ๚\ฟ๗?เ>ธv?6ๅ0<dmฟdฤ=ฟ?๖O>๗ไk?ู`?ck>EFฟXบtฟnฟ+	ฟ?<=ค?x`?~แ?แ k?iq/?ชฟป>์?ต<!{พช'ฟtHฟ%ๆjฟข|ฟุlฟๆ?uฟx,cฟดIฟข+ฟ้)ฟดwำพพใG!พฝ=ึ4>(ซ>ซiถ>฿ฬ?>ใ ?=?๖,?ี*?B]5?M๋>?สMG?๓ฅN?<U?KฎZ?_?ยึc?Yg?nศj?m?p?n'r?ว?s?โu??๚v?ศ.x?ำ9y?`!z?๊z?{?่.|?ฑ|?ิ"}?๑}?๑ู}?#~?_c~??~?}ส~?๒๓~???๔6?ไQ?8i?l}?๋??2ซ?ถ?gภ?ํศ?Nะ?ณึ?<??แ?.ๅ?ฦ่?ใ๋?๎?๋๐?๐๒?ฐ๔?4๖?๗?จ๘?ค๙?~๚?;๛?฿๛?l??็??Q??ญ?????B??~??ฒ???????'??D??]??(กjฟ{?fxฟ3I'?้ข=็\ฟtuW?ํ{>ิฟ3N<ฺ??q:>ฌmhฟเ2ฟuน>?จ}?8เ'?ฏ_Rพm๒_ฟuฟxฟ๒>iฃ7?ฟ8|?=m?a?6?;>ี?พr!ฟeท`ฟณ}ฟ*๒zฟ_ฟ1ฟl๓พะUvพ?vผ+V>ทฮ>L?ๅ?3?-LO?ฅธc?7๛q?[แz?๚;??า?๏]}?Cx?ศq?ฎi?+`?พ?V?oบL?แkB?78?์้-?x๑#?ปD?<๑?5 ?ำ๎?>ืณ๎>YQ฿>๚ละ>็ร>w#ถ>ฒ?ฉ>บ>์>๋>จ>o>พ@_>W๚O>GธA>กj4>-(>gp>ง>>Iz?=h๋=ซฦฺ=ฃห=๕ฝ=วjฐ=52ค=}ั=็9=ฒ]=`v=ืIe=fbU=โF=฿อ8=?๛+=# =ิ๐=ซ
=? =c๐<ฑf฿<๒ไฯ<ฃvม<ญด<Lง<๔็<@<ู<ีF{<ีi<JY<?}J<&o<<"Z/<-#<qู<N<"<ผ๔;Mพใ;๎ำ;ะ7ล;ท;ถศช;Lํ;ไ;เสฬพุ?=>ฮu>)ศAฟT0?%หฟi@
ฟง/x?ฉฺf=ฯ๚ฟ&6ฝธ{? ึ>่"7ฟnฟ"
พcEA?z? ๘>ัพพุ:\ฟ!฿|ฟฆ\2ฟ+H/พbภ> I?จง{?Rw?'ฌF?tC๕>>NxJพw๛พKZ8ฟa7aฟv{xฟ์?ฟVzฟtZjฟลRฟ	6ฟV6ฟ)๊้พ๘งพ~ฤKพ=Hฝศ=\}>@v>๚Aจ>๗ั>ฒ๕>า.?ฒ?ะ&?๙เ1?9?;?2D?LL?S?๖โX?{^?อyb?]f?2ภi?โฑl?๚@o?zq?Khs?Iu?v?ฬw?ไx?{ืy?ชz?`{?ส?{?ๅ|?ณ?|?ฃe}?าพ}?~?O~??~?8ป~?นๆ~?f?-?LI?วa?๚v?V?=?ง?๏ฒ?Cฝ?5ฦ?๔อ?ฉิ?xฺ?฿??ใ??็?ๅ๊?นํ?,๐?K๒?!๔?น๕?๗?K๘?S๙?8๚??๚?ช๛????ภ??/????ไ??,??k??ก??ะ??๙????;??U??1๏Tฟ<<?นไพCฟฝิ<?Lช|ฟฐ>yช@?Pฟ๔C๚พรA^??๚%(ฟ๊Pkฟnณ;ำคc?้ง[?ญฒ=y6ฟ๑ะฟ๛2ฟ๑Xฝๅh?ฒNo?ฮz?ผX=?Dช>ฌ๗ฝ[ฟ@ฯOฟ฿ึwฟfํ~ฟฺjฟJ?Aฟะปฟงลขพฑฉฝ|ำ>%4ฏ>M?ืz)?$(G?ย]?ซ๒m?|x?๗B~????w~?ORz?๙&t?ๅtl?ิงc?$Z??P?ืฬE?ฟx;?	:1?,'?ee?๓?แ
?*6?ล๊๓>~Aไ>pี>wsว>~Fบ>โญ>!@ข>ีV> >ฝ>F4u>มzd>จ?T>=FF>`ช8>๘+>." >>ะ
>:>j๐=โ๖฿=xะ=ึ
ย=Qด=หจ=(s=[=U=ไ;|=ๅฝj=ฮuZ=MNK=3==เ0='?#=>|=)็=๒=3ว๕<ธไ<ิืิ<ฦ<ุQธ<Sซ<<x<ฒ9<&ก<;fo<ว^<)PO<ฅ๋@<่3<0'<๙v<โซ<?<ธ๚;m*้;N๚ุ;ไ้ษ;5ๅป;ฌูฎ;?ตข;j;่?-ฟe?เ~ฟฆ?,?oพ$>๒epฟ์(?{?YU_ฟๆ?พf่M?ถA?๑ษพ?ฟI7๊พ*{??ใ๘3?ต1ฝยg7ฟyฟ{TฟR?ตพคึR>@J,?yq??~?ฬ_Z?อ?>>ฤHปฝ5อพพ'ฟุUฟ๕ทrฟฆฟXv}ฟpฟu)[ฟถ??ฟ{ู ฟฅฮ?พ?ผพZไuพLD๎ฝq:ช๑฿=6{V>?๔>์4ฤ>+๊>?$?ั"?T.?๐ป8?ัเA?ฎๅI?ิ๊P?ยW?.i\?a?'e?๛ฑh?ํฦk?หtn?ผศp?Nฮr?ขt?v?็gw?[x?ึy?yhz?พ'{?อ{?3]|?ดู|?E}?ฃ}?๔}?.:~?๒v~?ซ~?-ู~?ซ ?฿"?@?(Z?`p??I?นข?9ฏ?บ?lร?ห?า?ฉุ?๏??โ?tๆ?เ้?ื์?i๏?ข๑?๓?:๕?ฌ๖?์๗?๙?๑๙?ม๚?u๛???????r??ส????W????ม??์????1??M??ก<Xีฝ,๒ศ>@uHฟ??*#'ฟHพๆR~?wยพ฿?[ฟ๙?'่b?ะพ๛ฟH8ดพx/?งy?ฏพ>{6?พ  zฟbฐXฟ?แพIOย>JฎY?bใ?$W?ะ๒>dศด<ศมฮพํด;ฟ็ฬnฟป๗ฟ2sฟ@ซPฟณฟ๓Qษพ!พd=>ฌฆํ>m?x>?๋DW?mi?ฒu?ปๅ|?ี?โD?Gไ{?ผOv?ฺo?ฅf?$7]?]OS?sI?มฤ>?อ{4?\*?| ?๎?xผ?ว๏?`๙>ใ*้>ฝฺ>ิห>๑eพ>ยฑ>2ใฅ>ฌฟ>lO>>สz>)ณi>ญปY>)ำJ>H้<>[๎/>hำ#>1>2>ค7>๖*๖=น&ๅ=<Mี=yฦ=ชหธ=9?ซ=ณ?=ถ?=โจ=ฬ=ื1p=!_=งP=A=u*4= ช'=ก=ก3=ฺ=๛}๛<V	๊<ฑสู<ฌส<?ผ<Wฏ<GTฃ<ฎ?<p<แ<c๗t<ม๕c<T"T<$hE<ฌณ7<ย๒*<<)	<ย	<ณ1 <๎;?;๗ฮ;ิCภ;ก๊ฒ;ญ~ฆ;~๏;ป๕?บ~ฟขuk?9ฟMDผ๘่A?นฎtฟ๚้=พอl?v้ฟศEXฟ2ํ>ด๒t?์Iผๅoฟ๗I:ฟ๏b>Im?T^?Kf\>ปPฟฏuฟูlฟำนฟ๒<ํพ
?a?
๐?ส2j?B.??ธ>ี <ผZพ!Mฟj฿Hฟ-akฟHะ|ฟ>ZฟนuฟภbฟIฟไ+ฟ๏
ฟษ8าพ%สพ8๑พ๙Lฝiฆ=Dท6>฿>7/ท>/~?>ื ?e?Gk?ูถ*?ฒ5?ู??sG?ฦN?ร.U?1วZ?ฝจ_?ซ้c?ฮg?ฟึj?	คm?ap?า0r?๎t?๕u?? w?4x?q>y?`%z?ํz?{?1|?ูณ|?ษ$}?ข}?i?}?ฺ$~?yd~?~?Qห~?ช๔~?y?~7?[R?i?ล}?8?U?lซ?มถ?ภ?ษ?oะ?ฯึ?T??แ?@ๅ?ึ่?๑๋?ก๎?๕๐?๙๒?ธ๔?;๖?๗?ญ๘?จ๙?๚?>๛?แ๛?o??้??S??ฏ?????C????ณ??฿????'??D??ะิY?0!\ฟst?bฏ}ฟวั9???]ผ้Oฟผๆa?}โ/>๚ฟ ษ=ๅๅ?h{>ฌบmฟ?ผ)ฟ์้ฮ>Kไ~?c' ?ก?vพv	dฟ๐ฅrฟ?๑๗พ๘D;>!<?ไ0}?k?ั5?ร(>ฅพท$ฟbฟ~ฟj?zฟ๔]ฟ?z/ฟฟ๎พ?^lพ๘ฑ:ฤื^>0ิั>/เ?fC5?EP?ภld?ฌrr?ฯ${?ๆS?Kว?ฤ5}?ๆAx?์xq?"Si?L5`?rV?YNL?ํ?A?๔ฎ7?-?แ#?เ???(ค?y??>d๎>อณ?>91ะ>ฝย>?ต>เฉ>&>I>0>a/>๎้n>`^>_O>S'A>iใ3>'>ุ๚>๙9>u4>	ผ๛=/V๊=!ฺ=?	ห=ฯ๛ผ=~ๅฏ=ถฃ=๖]=[ฮ=๙=ญฅu=]d=๎ภT=?E=?A8=y+=?==ผ1
=^ =Z๏<ฝ?<Gฯ<#ไภ<Xณ<m
ง<ใq<`ง<<z<๚#i<~๔X<ขไI<pเ;<Uี.<ฒ"<of<ใ<<ญ๔;ัใ;
Nำ;rขฤ;๛ถ;]Gช;๎t;{?นฏฟ๊>ดe	>ฃ0ฟ?๙?D\ฟฟ?๐พ2|?dํ\ผบ:ฟ2็<๗๑}?ล๔ฝ>3ฃ?ฟํ)jฟ[พฝwทG?๙rx?kฎ่>?0ฃพล๚_ฟฟฎ{ฟV-ฟQGพUาส>FWL?๊|?3?u?Z๘C?๔M๎>๙๛=?ฺWพ?D ฟ๋d:ฟObฟ"yฟ๑?ฟV?yฟฎiฟภฌQฟkฦ4ฟt?ฟ'็พฉ`คพคFพ์RฝโZ*=ฤ>ง๏y>Xช>ฦซา>ํ๗>พะ?คA?ญ	'?ยP2?|=<?c๔D?}L?FS?Y?ป4^?ฅb?ดf?[แi?ถฮl?Zo?ูq?/{s?ฎ%u?ฦv?ืุw?L๏x?ยเy?ฒz?g{?ี|?!|?<}?i}?:ย}?	~?Q~?6~?#ฝ~?b่~?ึ?G.?`J?ถb?ษw?
?ุ?ง?dณ?จฝ?ฦ??ฮ?๊ิ?ฐฺ?ฑ฿?ไ?ล็?๋?ิํ?D๐?`๒?3๔?ศ๕?'๗?V๘?]๙?A๚?๛?ฑ๛?E??ล??4????็??/??m??ฃ??า??๚????<???f?7Nlฟ9JX?ุๅฟ]=Sา!?้ฟS๎๊>คฉ+?r_ฟIษหพฉ;i?z?/V7ฟe๛bฟฏ๊=oฺj?,2S?"ซ๘<wX?ฟฎฟ?)ฟฏผบ?r?x?iบ7?่>ฆ?พฒ1ฟฑWSฟ6yฟb~ฟ฿hฟz๑>ฟ 	ฟ-พฝ>=ต>฿??<+?฿ฦH?b๔^?~ฤn?ๅ y?	{~???qF~??y?ฃตs?ึ๎k?-c?๏zY?2nO?๗&E?=ำ:?แ0?R&?ห?ํ^?/S
?rฎ??่๒>+Mใ>;ิ>อฦ>ฆyน>"ญ>ก>ญ>>t๘>t>นwc>ศ้S>}dE>ยื7>-4+>
k>an>๘0
>Pฆ >A๏=จ๕?= ฯ=ภ+ม=หณ=dWง=ฟ=ฝ๓=L็=g{=ฏi=#zY=dJ=}Y<=๗G/=O#=~ฬ=C=บu=เซ๔<[ฐใ<sโำ<C-ล<V}ท<ภช<ๆ<5?<S<ื<2Rn<งฦ]<aN<3@<ๆท2<O&<ตร<<b<ฬn๙;่; ุ;ษ;ป;ฎ;]๚ก;mF?พ๖๎ฤ>๒ฟฬN?ฟ'^F?wrุ<งucฟd๊=?"?๙>3ูjฟฆำพ๐rZ?ฌ2?xล์พอAฟรหพฎ?อแ?9*?ูxฒฝi?ฟ๑๐ฟท0Nฟคพf8u>๔D2?ๆs??)}?โูV?6w?่@j>9ฐๅฝhAึพั*ฟ-Xฟ๕sฟกfฟ๔|ฟloฟ4Yฟ?>ฟาฟฺ๛พ\ฌธพะรmพบฬ?ฝปด๙;~Vํ=)\>ึผ>Wถฦ>/j์>ช?Y?ไL#?b/?์W9?ปiB?ก]J?ฟSQ?giW?#น\?ฬZa?ฒce?ศๆh?ุ๔k?ณn?f๋p?g์r?ยฉt?@,v?{w?ix?y?Kuz??2{?(ื{?e|?๐เ|?ฺK}?{จ}?ป๘}?A>~?zz~?คฎ~?ำ?~?๖??$?8B?ฅ[?ซq?ฝ?A?ฃ?๒ฏ?ญบ?๗ร?ฬ?๛า?ู?=??รโ?ฎๆ?๊?ํ?๏?ร๑?ฌ๓?S๕?ม๖??๗?๙??๙?อ๚?๛????????x??ฯ????[????ฤ??๏????3??่?=Cษxพ]G>๚ย2>v"ฟุt{?^ปGฟ๕Mพ4|v?ฟ๒Gฟฒ|#?^2S?ๆ$สพศJ~ฟฺiพD??ดt?4\>ฏKฟA!}ฟฤOฟ9?fพง?>G-`?A?ุๅP?ฅ฿>ฟผ฿พJAฟ"~qฟ๘๚ฟนtqฟฤ๙LฟuฟOKฟพ๘พ?ท=ฅ?>ีิ๔>ีb!?ศฮ@?ิY?ฃฉj?ีzv?NK}?;้??ภ{?รu?Rcn?^ะe?>h\?ฆwR?<H?f่=?ีก3?ง)?Kญ?'&?ร??ป8?%ฝ๗>ถเ็>?ุ>ฎส>Pฝ>?ฝฐ>s๎ค>4ฺ>x>xภ>hRy>ฐSh>nsX>ย?I>bห;>ฐใ.>ยฺ">gข>-->\n>ํณ๔=cษใ=แิ=|[ล=ฑท=๘ช=# =	=๑ิ=F=ยn=D3^=vษN=๎p@=ิ3=ฉ&=โ=pU=Q=?๙<(ฃ่<]}ุ<`vษ<Rzป<ฒvฎ<GZข<<
<hี<hs<ฮb<?R<๕9D<w6<ํ)<๛ <&<9๏<ฺ๋?;Q)ํ;/ฒ?;ฏ_อ;ฟ;ฝุฑ;อฅ;ณ	~ฟaTx?N{}ฟจ|?'ุEฟฤ@>" ?;๋}ฟXJ>ธเ[?6?ฟU?Dฟมญ?๖3k?,์ฝ (wฟข?)ฟฌh>?t?FฯT?ย็>ผฟึqyฟeAgฟ;๗พ๑1=๚?RKf??๗?mf?ัฺ'?ถ่ฉ>า๙Jผ1ชพ]ฟญoLฟ๗umฟ}ฟด๙~ฟฤutฟี`ฟๆธFฟdh(ฟ๑ฦฟ คฬพ"Tพ=}พ?ํืผ ็ฌ=ร?>=U>฿บ>iแ>ฟ7??ผ?ื?ุซ+?^e6?Fำ??"H?ๆWO?ผญU?6[?Z	`?า=d?็g?rk?h?m?{Cp?Zr?1+t?mพu?Kw?ศKx?๙Ry?-7z??z?}จ{?=|?ไฝ|?}-}?-}?๑แ}?*~?`i~?ิ~??ฮ~?ู๗~?<?แ9?mT?jk?R???nฌ?กท?Tม?บษ? ั?Mื?ม??zแ?ๅ?้?.์?ึ๎?#๑?!๓??๔?Y๖?ค๗?ร๘?ป๙?๚?M๛?๎๛?z??๓??[??ถ????I????ถ??ใ??	??*??VฌDฟ?!?ึ฿)ฟ^หT?ฬน}ฟ_?Z๛xพT+ฟeu?ขงkผS{ฟ?๔r>ะ้{?G\ผ&xฟFฟฬ3?9๛??฿?Bจพีmฟ_wkฟ/oึพ>kฦF?ฬ๛~?e?m?w๐=MคพYฉ,ฟ์?fฟฅฟฏPxฟฮxYฟR
*ฟYVโพ"tSพ.ษ<F_t>ฦ+?>Qล?ชa8?3งR?๗#f?s?ล{?ท?Pค?ทฬ|?Sขw?-ฐp?zlh?:_?ajU?w?K?0๎@?0?6?ฑu,?""?ๆ?ว??พ??>Rn์>+?>iพฮ>b$ม>Wด>QOจ>0>'p>g>
~>?.m>๑๛\>?M>Dพ?>2>?I&>ึ>)>(6>1โ๙=ส่=~ุ=ษ=Lป=ฎ=ข=>>=ย=A =ีs=R์b=อ.S=SD=งๅ6=?4*=Ae=Cg=e,	=PN?<๐ํ<B?<yฟอ<Kwฟ<ั,ฒ<vฮฅ<ฺK<ภ<๘<ฎx<๔jg<ZW<ถfH<}:<-<@~!<H<ู<#<5๒;Adแ;Lพั;u.ร;mกต;<ฉ;เ#ฟจ9F?P?ฟQ?ฐ0พ๙พ=Qx?๐p>ฟ_ืพ8๙?uแ>พงฐxฟ๛<6>๚?|๑|>ดษRฟ]ฟC?E<hV?ฅฮq?>ม>๊ศพRwhฟรๆwฟ.R!ฟfQถฝ*_ไ>o๗S?r:~?้ur?=?)ป?>ซ2ณ=>yพฟ^?ฟฎ?eฟ{zฟ<์ฟwxฟ์[gฟFไNฟ๒1ฟxฟ~>เพึพna9พwฝ?X=i!> า>bjฎ>*ึ>2ฅ๚>ฃb?แฅ?hD(?f3?+1=?สE?RM?ฆ้S?eซY?Hฑ^?c?+โf?3j?(m?o?ฤลq??ฉs?NNu?ปv?i๗w?ฯ	y?พ๗y?ฦz?ิx{?อ|?|?y}?Ns}?ชส}?X~?่W~?ฒ~?ใม~?์~?g?]1?M?e?สy?ฦ?Y?ึจ?ด?ขพ?eว?๛ฮ?ี?=??*เ?oไ?!่?T๋?๎?๐?๒?`๔?๏๕?H๗?s๘?v๙?W๚?๛?ม๛?S??ั??>????๏??6??s??จ??ึ?????!??")tฟุฝ?ไ๑ฟz?B฿Pฟ1ถ>)ษ>ฑ?xฟ*บ(?Bv๙>OtฟYพ~x?]lฝ>ฉ Qฟย9OฟเDX>อ฿u?bฅ@?.ฃฝปOฟบพ{ฟ๛]ฟ้$z=?'?ฟw?3ํs?/Y,?!g~>J	NพึปฟัYฟ{ฟ}ฟVdฟงไ8ฟฺฟffพ5W%ฝแ?7>Uัภ>!พ?ฃ/?ฺุK?5a?bLp??่y?ซ?~?)๑?ฅเ}?Qy?ีr?!็j?๐a?FX?Z/N?[ไC?ฟ9?DZ/?nX%?๛?"??ซ?	?อง ?เ๕๐>tแ>ฬสา>๗๔ฤ>ชํท>ฎซ>z.?>pf>;M>๒ู>]r>Ka>R>eฐC>๊@6>ฟธ)>H	>ฅ$>ฒ?>
?=?oํ=ึ?=Oบอ=ใ|ฟ=i:ฒ=?แฅ=[c=ฐ=๑น=`่x=Mฅg=W=ชH=oด:=ภ-=ฑ!=y=ต=พO=ฒ๒<"ณแ<า<Atร<๎โต<ฃBฉ<ช<t<f<ะ?}<=l<ึ[<vL<_><(1<?$<i<ๅย<ู<ั@๗;Rๆ;๊ึ;j?ว;jน;ซฌ;D็>?7ฝ์๚ฉ<มำ[พุ?I>oฟ์ik?8
nพ<@ฟฺ_?H>{.zฟ
+vพเีm?bง?พOฟฑ9zฟพฝ(?แ/??์9พ	ธMฟญฟT๐Aฟb๖พa>$K=?D๙w?ดรz?บคO?ฺ??>>wคพ๙~็พ1ฟ๚\ฟี/vฟสฟีึ{ฟ'mฟฅ~Vฟถ[:ฟศ?ฟ๋r๓พoฐพX^พไภฝPฎ<y>1lh>ๆข>ห>จป๐>(๙?^ผ?Xฯ$?Z0?:?ดpC?๛CK?4R?ZX?กR]?เa?,ุe?!Li?๗Ll?H้n?์-q?)&s?ใ?t?ภWv?Mกw?"ฟx??ถy?ไz?-H{?ฃ้{?u|?ะ๎|?เW}?ๆฒ}?ย~?F~??~?ด~?่เ~?]?ฌ(?E?^?$t?โ??+ฅ?Wฑ?โป?ล?๊ฬ?รำ?ฑู?ำ??Fใ?็?u๊?Xํ?ุ๏?๒?ใ๓?๕?๊๖?"๘?0๙?๚?ไ๚?๛?+??ฏ??!????ุ??"??b????ส??๓?????Uพ?3ศ!ฟN?CฺOพ฿ูงพv^?]ดnฟa >ย[?ี4ฟษRฟฐI?hw2?ํฟzuฟโoำฝๅ?V?g?f2>ิล&ฟsะฟG?ฟiปพ)?lฝi?ๅ๔|?tE?ภ>?ฝv๙พฬ"Jฟ|uฟ๊ฟ[~mฟษFฟd5ฟฦcฎพคIืฝLe๕=ึฺฅ> U ?=&?[D?เ[?oฆl?ทw?Iๅ}???Kผ~?่ะz?ัt?๘?m?ํd?l
[?๘Q?ซสF?Kv<?142?	!(?TR?บื?'ผ?ฟ?Dw๕>.นๅ>$ำึ>Fยศ>ป>ธ
ฏ>	Vฃ>T[>๎>๖p>฿v>v	f>๙OV>ภกG>๎9>?&-><!>ๅ>๘ฤ>ผ>B๒=่แ=d้ั=Lbร=!?ต=Bฉ=`=n=s=$๛}=3^l=K๙[=๕ถL=,>=IK1=่?$=ุ=?โ=Q๘=n{๗<?Mๆ<กQึ<3qว<น<ฯถฌ<yน?<(</<<<q<S`<6ภP<%BB<กล4<ษ8(<๛<ปฌ<ข<L?;dศ๊;{ฺ;^Pห;ฬ2ฝ;ฐ;w?๘๖SฟifF?oZฟทซz?,ฺqฟZ?>ฐ?ธ>7ื|ฟชN?ฮ45?ผ+Iฟf?ฟg7?T?Iพอก~ฟฟm?>ฟ|?E9B?๓ =นS*ฟื}ฟp\ฟiาะพ3m>mฝ"?โPm?26?Z_?แ?,<>ฏsฝข!ฟพR!ฟ?#Rฟ_ฑpฟๅ~ฟึ'~ฟ@2rฟฎ]ฟ]ฎBฟ๐#ฟDฟpEรพผ/พ๋พิr)ผMห=M>yช>Vภ>ดๆ>ใ?ฌฤ?๑L!?ึA-?ซส7?A?,I?zHP?๏V?oํ[?@ฉ`?ษd?>`h?ูk?7n?p?ตr?/gt?|๒u?uIw?๓rx?๐ty?Tz?{?พ{?GP|?ฮ|?โ;}?ฅ}?ฟ์}?฿3~?{q~?ฺฆ~?ี~??~?ฬ?ี=?ูW?an?ไ?ส?lก?ฎ?น?ย?ฯส?๐ั?ุ?u??โ?ๆ?้?์?.๏?o๑?c๓?๕?๖?ฯ๗?่๘??๙?ฎ๚?e๛???????i??ย????Q????ฝ??่????$+?wจพ-99>;Cพชา?๔ต[ฟศ}?ถ฿ฟตลษพY๘?hพํjfฟ๎>๏k?"eพงฟlะพ1๎$?ขA|?Fiี>ใ๔่พzwฟฆิ]ฟ~กพQ;ฒ>๔XU????ขิZ?ถi?>คณ:=tฤพ8ฟ1mฟ+?ฟu?tฟ้Rฟ!ฟืฯพญ-พ?Nt=^>?#้>??<?'V?5ฃh?1u?ูข|?Mฦ?z_? |?Hฅv?งvo?Gg? ท]?ีS?ใ?I?กM??L5?อเ*?x? ?sj??3?ัa?^๒๙>ฌ๘้>\ืฺ>>ฬ>แฟ>+eฒ>ี{ฆ>หN>zี>>(ต{>jj>sZ>QK>ฆ=>บ0>n$>ฯ>๚>;ต>๗๗=ฑ ๆ=?ึ=Gว=ด{น=!ฃฌ=Mญ?=ฦ="-=็=q=q^`=2ฮP=฿QB=rึ4=2J(==Eพ=แ?=$n?<ี่๊<ฏฺ<#nห< Oฝ<๘*ฐ<F๐ฃ<ฺ<ข๗<<^แu<ฯd<๔์T<ณ$F<$c8<+<tฌ<<ฑE
<'ฌ <uz๏;$ฺ?;Raฯ;|๛ภ;ณ;ั?=?ฟพqฟ๘ฦ{?zwฟ฿ุZ?ฒcฟ"พ๑พS?dHkฟำaz< รt?D฿พtbฟหซส>7y?QU=yษiฟdสCฟ{.>ณh?x๗c?+>7?พฟ๊rฟ??oฟํ|ฟกิบ9ฆู?พq^?โป?หgl?๖?1?Lม>N	=UพฟFฟ๛jฟxL|ฟRฟnyvฟw๋cฟ?Jฟ3ญ,ฟeCฟyฐีพ30พำt%พ฿ึ+ฝA=y{1>*$>uต>ห?>?๔?>*ฟ?|ฝ?i*?5?.>?G?kN?1฿T?ฝZ?/l_?๋ดc?ๅog?ะฎj?Sm?8๕o?กr?2๐s?:u?แ๏v?A%x?1y?7z?็ใz?บ{??*|?ญ|?T}?่}?Pื}?N!~?fa~?๋~?ษ~?ช๒~?ฟ??5?Q?h?ฬ|?a??สช?5ถ?ภ?ฉศ?ะ?ึ???แเ?ๅ?ฉ่?ส๋?๎?ุ๐?เ๒?ฃ๔?(๖?z๗?๘?๙?w๚?5๛?ฺ๛?h??ใ??N??ซ??๛??@??|??ฐ????????|?0tฟ"ษ]?dต_ฟ5Zv?ขฦ|ฟฐส5?~J๕;รหRฟซ_?๕@>ิ?ฟแห=๙?Bb>ฬlฟeึ+ฟส>Cง~?O๋!?Yจnพ๓"cฟซ9sฟN๋๚พฌ5>B ;??|?ak?9(?P๎,>ฏชพ๛#ฟ*bฟก๓}ฟฤhzฟุ็]ฟD๗/ฟKฉ๏พLnพ<Dบหโ\>N๚ะ>#?V๚4?P?GDd?เWr?ผ{?ขN?ส?๏>}?Px??q?ีgi?ไK`?PV?ษfL?B?kว7?-?Hก#?4๗?บฆ?๒ธ?g?>๛2๎>aื?>ะRะ>cกย>eฝต>ีฉ>ฯ@>?>0>ไD>$o>๏ฟ^>O>HA>๐4>?'>c>ตR>K>?ๆ๛=1~๊=?Fฺ=,ห= ฝ=ฐ=!าฃ=x=ขๆ=/=มฯu=รd=bๅT= F=a8=t+=Vฎ==lI
=jฐ =ง๏<นใ?<kฯ<6ม<ณ<'ง<<.ภ<ฐณ<ณz<๕Ki<ฒY<AJ<ง <<P๓.<ํอ"<f<ฟ๛<2<,๔;ม8ใ;Frำ;+ฤฤ;๘ท;็&พธพ1ถ?>่๋๘พิ;>">]>4ฟ*??CฟV๙พ๒i{?;F~ฟ?กn<_}?-ร>็ม=ฟ6kฟษัฝยIF?;๓x?-์>ึทพb&_ฟต๗{ฟณ.ฟดพVyศ>sK?ตR|?&v?ฝD?kแ๏>G>"ึTพK?พ}๏9ฟ๕8bฟล๓xฟ๛?ฟ"๙yฟดiฟj์Qฟ5ฟฎ*ฟFว็พm?คพาฑGพฝฅ)&=^อ>y>E?ฉ>?Oา>ูห๖>7ฌ?A!?ํ&?72?R'<?๓เD?xL?.7S?Y?f)^?ตb?{f?แูi?6ศl?aTo?๑q?์vs??!u?v?ึw?โ์x?ช?y?Hฐz??e{?x|?๓|?6}?ฎh}?vม}?_~??P~?ถ~?ดผ~?่~???-?"J?b?w?แ?ต?jง?Iณ?ฝ?xฦ?.ฮ??ิ?คฺ?ฆ฿??ใ?ฝ็??๊?ฮํ??๐?[๒?/๔?ล๕?$๗?T๘?[๙??๚?๛?ฏ๛?C??ฤ??3????ๆ??.??m??ฃ??า??๚??ดถส>Oฟ.q?๑ดtฟธc?ฉ'ฟ_0>?ัศฟ:?ๅG ?็eฟไลณพ๑m?๗>ow>ฟT^ฟ฿ๅใ=ตn?ฟฌN?ษY๚:สวCฟ.f~ฟF%ฟฒาืธปส?\๛s?ๆuw?Mฺ4?nC>T&พฯ๓ฟ{Uฟูyฟฉ~ฟ๕ugฟวf=ฟPฟด^พ nzฝV$>I>ธ>น ?	,?้I?`_?w+o?g>y?~?ํ๛?#-~?ัy?O|s?Iซk?rศb?+Y?"O?฿ำD?b:?=E0?ฅ=&?ใ}?ฆ?
?ขj?h๒>ำโ>์ิ>-ฦ>Zน> ยฌ>W1ก>	Y>]0>lฎ>s>h๖b>qS>ห๓D>n7>า*>>*>แ	>S\ >g๛๎=Au?=lฯ=fผภ=้cณ=?๖ฆ=:e=?=i=hz=(i=?X=#๏I=ช์;=ฏโ.=ภ"=ฤt=๔๑=ฟ)=t๔<ฟ,ใ<๙gำ<Hปฤ<Dท<?]ช<:<น<ฦJ<<jศm<oF]<อ้M<*?<P2<f๏%<;j<อฑ<ๅท<?๘;^็;:ื;ฺศ;f?บ;kฟ๊?C3ฌพุd>8์้พีtA?|}ฟvQ?t(ฝHค[ฟaG??5แ>`ณoฟๅใผพฮ`?+?ศั?พ๙h~ฟฝ9ผพg?แ??G๑$?ญไฝABCฟ  ฟ9#KฟพH#>/5?u?|?ํ	U?z๋?6_>vั๚ฝทฝฺพะ:,ฟ	QYฟjtฟeฟฎ|ฟภูnฟฮXฟย=ฟฮฟธ๙พUถพหณiพืฝeฑ6<๔=ช_>ช>๗๕ว>nํ>6?x?ฑ#?ซ]/?ฅ9?แญB?UJ?๖Q?W?ํเ\?y}a?ไe?i?ฐl?ฐn?ฅ?p?a๛r?มถt?7v?Zw?ๅฆx?๛กy?ฌ{z?b8{?๓?{?ณi|?ไ|?๘N}?/ซ}?๛}?H@~?;|~?)ฐ~?$?~??ู%?C?c\?Or?K?ฝ?๚ฃ?Oฐ??บ?=ฤ??ฬ?/ำ?0ู?d??ๅโ?ฬๆ?-๊?ํ?ข๏?ิ๑?บ๓?_๕?ฬ๖?๘?๙?๚?ำ๚?๛???ค????{??ั????]????ฦ??๐??อฟ&oผ<ต>ฏlๆพแึญ>ผ2ผ๙พp?๙?]ฟ๛<๙k?~fฟซ4ฟผ7?คC?ษ๕พkลzฟ:พIภK?๐n?บur>ุํฟ๛แ~ฟ?กGฟว=5พg๐>p4e?ฝI~?IVK?๘วฯ>ถ์Aฝ์พๅำEฟํsฟีฟ~oฟยลIฟี"ฟgสถพp๘ฝฯึ=#>_ำ๚>ู#?๗พB?[zZ?kญk?ew?m}?๕?๊~?){?ณJu?]ัm?,e?qธ[?ธภQ?dG?.=?ข้2?pา(?g??}?[?k?จ๖>สๆ>ีื>๒ตษ>gผ>Nโฏ>^ ค> >ร>)>สx>ุ+g>%_W>?H>ยฺ:>A.>~	">U฿>aw>๘ฤ>Px๓=gฃโ=๖า=\ฤ=ฤถ={ช=UR=qY="=๙@=}m=]=ดฝM=ธw?=ใ.2=ผั%=?O=x=ฃ=<น๘<มu็<฿dื<Xqศ<gบ<ฃญ<่ก<CQ<?แ<?+<?Dr<+sa<YฬQ<ฌ;C<ีญ5<?)<T<ฺg<ฤ=<ฆ?;๚๕๋;-?;Uฬ;ี%พ;ฆMTฟ๙?ๆloฟ๐d?ูสpฟV๗?O_ฟฎิฑ>๛?>l๗ฟตะส>ิไI?้5ฟผB2ฟ%?fม`?ๅ็Mพผ{ฟ?ฟs.ผ>kธx?โหK?!ฟ=NB ฟR๕{ฟbฟไพ
t์=?ม๘i?ถ?ํc?;{"?>๋*ฝuฤดพPฟNZOฟ$!oฟ~ฟi~ฟWsฟ-_ฟฒDฟ?)&ฟ่mฟ๊๋วพWธพGฎพ&ถผq@ผ=ท!F>L>ฝ>^+ไ>_?ฉร?|i ?%y,?7?๓q@?5คH?ัO?V?ฬ[?>Z`?Ed?\$h?ลKk?ม	n?ภkp?}r?It?ฤุu?%3w?_x?(dy?Fz?ํ	{?ฎณ{?ฮF|?Lฦ|?ล4}?|}?i็}??/~?zm~?aฃ~?า~?๚~??แ;?(V?๊l??ฑ?y??Fญ?\ธ?๖ม?Fส?zั?ถื???ษแ?ึๅ?X้?a์?๏?I๑?B๓?๗๔?r๖?บ๗?ึ๘?ฬ๙??๚?Y๛?๘๛???๚??b??ผ??
??M????บ??ๆ??๔๏ฟK?ฺS๗พFช>LษพH!?jฟyx?sฌ๏พ\จ๘พ0?~Tพ?อnฟ2ห>'ดq?ฅ"พWX~ฟ|๋พฦ>?~?Rฒ๊>ค ีพ฿งtฟโobฟTฒพ9งข>ิ Q?็?๙5^?8?d=หบพJ4ฟน6kฟผฌฟvฟgUฟ:$ฟุpีพ๋`9พuG=T>อไ>เฤ?a;?8U?ํ?g?Wณt?R`|?rต?w?ฤW|?ศ๕v?ฬูo??wg?ั0^?RTT?ํ"J?.ะ??5?_+?ฆx!?Xแ?๛ฅ?ีฮ?ืม๚>uฝ๊>?>ๅ;อ>Oธฟ>ต ณ>?ง>บื>ฟU>>ฌ|>:`k>iL[>@IL>XF>>๒31>%>7ฅ>๘>m->๋๔๗=Nัๆ=ฺึ=w?ว=0$บ=@ญ=Z?ก=ฟ=ตซ=บ?=]๒q=*a=:Q=ฝC={5=fใ(=/+=๘B=_=?S?<ฟพ๋<มa?<f'ฬ<๛ฝ<iหฐ<ค<ห<๏x<์<Qมv<ๆe<ไฎU<-ูF<9<V2,<ไ= <่<ฃร
<[!<T๐;!ฅ฿;7ะ;Cซม;,Cต<S่?n&`ฟฐmq?qkฟ;ฮF?๖-อพm์|พ8b?+ศ_ฟ)ฃฝpmz?ธพิ๙jฟbดจ>ม|?^~่=kcฟPLฟJ๘=#c?/ฤh?ฯน>wู๎พL๘oฟฝrฟmิฟู๗แผ";?>;P[?h?fin?bk5?	ส>[iN=ฺพ?ฟ0aDฟฯฑhฟ$ล{ฟ@ฒฟเ*wฟ5eฟ๋Kฟ9.ฟ?ๅฟA?ุพBoพEฏ+พ็QCฝฝW=?w,>ึ?>q๘ฒ>ญตฺ>)M?>ๅ?ถ?')?๎4?C->?,งF?N?ษT??Z?	2_?Dc?ฮCg?wj??_m?Aุo?z?q?[ฺs?Gxu?p฿v??x?3%y?~z?ฺz?ฌ{?D#|?ง|?}?]}}?aำ}?ๅ~?r^~?\~?หฦ~?ฟ๐~??4?ฯO?kg??{??็?.ช?ฏต?คฟ?Dศ?ผฯ?4ึ?ฮ??จเ??ไ?่?ฅ๋?`๎?ฝ๐?ศ๒?๔?๖?k๗?๘?๙?m๚?,๛?า๛?a?????I??ฆ??๗??=??y??ฎ?????^ฟ	Tv?น?zฟง i?ฯjฟLS{?2ืxฟ5(??ึ=?[ฟ4?W?ฎw>ลฃฟm~<ฎใ?tฝ7>ฒhฟws2ฟC}บ>จบ}?t'??Tพ?%`ฟ~?tฟ;5ฟn๋ >์ฺ7?gE|?#m?.?๒:>9Iพฑ!ฟvฮ`ฟื}ฟ็้zฟ_ฟ1ฟ๒0๓พU?uพf_๖ปV>฿;ฮ>d_?k4?XO?Hมc?๔ r?ไz?'=?vา?\}?P}x?Tฤq?4ชi?k`?ฝึV?JตL?ฑfB?8??ไ-?์#?๕??ฆ์?า๛?xๆ?>่ซ๎>ึI฿>ใพะ>9ร>.ถ>ห๙ฉ>1>๋ๆ>2ๆ>>o>ะ8_>๐๒O>^ฑA>/d4>*?'>ฬj>Qข>ฑ>7q?=๕?๊=อพฺ=Bห="ฝ=ldฐ=J,ค=๛ห=ฦ4=์X=+Wv=Ae=ณZU=ธF=3ว8=
๕+=] =t๋=ซ
=^๗ =ธ๐<ก^฿<p?ฯ<ฆoม<-ด<?ง<Sโ<<๙?<ร={<?ฬi<nY<ญvJ<Xh<<อS/<ธ'#<๕ำ<I<bz<2ณ๔;ถใ;ๆๆำ;ฑ0ล;ขlZ?gพณLพ?ำ>ฃฯพeหB>azp>%๛@ฟุF?รณฟ๔k	ฟลhx?:Y=๐๗ฟ	อ๐ผ[ุ{?oี>=7ฟjnฟพะA?|rz?็U๗><|พปi\ฟธั|ฟl#2ฟฟ#.พโภ><HI?ฒ{?ภ	w?|F?ฺ๎๔>h?>ใKพ"G๛พZs8ฟ\Gaฟ๛xฟ&?ฟํPzฟMPjฟ@ธRฟ.๚5ฟแ%ฟศ้พต๙ฆพPKพทฮฝAซ=ฑ>์7v>Wจ>`(ั>yร๕>6?ํธ??&?Oๆ1?๊฿;?OขD?OL?:S?ถๅX?แ^?ไ{b?h_f?ศมi?Bณl?,Bo?!{q?2is?u?;v?อw?ๅx?์ืy?pชz?๋`{??{?%|?๊?|?ำe}??พ}?:~?$O~?~?Pป~?อๆ~?x?-?YI?าa?w?_?D?	ง?๕ฒ?Hฝ?9ฦ?๗อ?ฌิ?zฺ?฿??ใ?ข็?ๆ๊?บํ?-๐?L๒?"๔?น๕?๗?K๘?T๙?9๚??๚?ซ๛????ภ??0????ไ??,??k??ก??ะ???P฿>ถ>hIฟ ฟm?5๛qฟ฿?_?Oึ!ฟ(+ๆ=ส_?q๖ฟ&?>;c$?ฐcฟ_[ผพDTl?i?>X๘;ฟ?`ฟา\ส=w๕l?MP?dำD<8Bฟถฃ~ฟุ$'ฟยฃ๛ป[?ัws?4?w?แ5?5ฺ>
!พบ๙ฟ0tTฟg?yฟี/~ฟฝีgฟg๓=ฟM๕ฟ?ธพ>zฝ๊!>-ท>~ญ?9,?>KI?ๆU_?o?ฎ(y?H~?ส??46~?แy?นs?Sรk?ใb?รGY?N9O?g๑D?ำ:?@b0?๔Y&?N?	/?S%
?น?ฦ๒>x?โ>เ>ิ>ดSฦ>ฐ7น>#ไฌ>_Qก>w>xL>ทศ>ฤลs>U$c>๊S>ัE>๖7>๒๔*>0>m7>ย?	>v >\,๏=?ข?=ใ;ฯ=๑ใภ=ผณ=$ง=%=ษฝ=ต=ๆปz=yXi= )Y=ฉJ=P<=ง/=แ"=๋=๓=ปD=ฎP๔<|[ใ<xำ<รใฤ<๐8ท<้ช<ูช<ง<g<3บ<Y๙m<๘s]<-N<ล?<Du2<&<<_ฯ<jำ<อ๙;ว็;ฏื;ถศ;ฌ]f?'/oฟ?{?ฝพง>,Z๘พ็ZF?ฬ`~ฟฒถM?๋ผื^ฟ?D?ฬ๊>nฟ๋ัฤพ?!^?sส-?<ฮ๗พJฟ~ฟฤมพฏะ?E๛?ฃฦ&?u?าฝ็Aฟ?ฟ ;LฟO;พ>ร'4?แขt?ะ|?ฝฏU?ำ?%c>3Q๓ฝ&ูพฌ+ฟฬ้XฟXtฟ?zฟว|ฟ.oฟพYฟ๙g=ฟธ*ฟ?A๚พฬRทพB%kพัูฝ๋&"<ฅ๑=บ^>ฐก>ว>ะ"ํ>๏^?^P?y#?U>/?9?ฑB?$J?nuQ?ำW?ฮา\?+qa?-we?ป๗h?l?ฉn?๖p?๖r?$ฒt?3v?แw?โฃx?^y?iyz?l6{??ฺ{?:h|?Bใ|??M}?9ช}?>๚}??~?{~?ฏ~?ญ?~?ณ?%?ลB?\?r???ิฃ?.ฐ?เบ?$ฤ?*ฬ?ำ? ู?V??ูโ?มๆ?$๊?ํ?๏?ฮ๑?ต๓?[๕?ศ๖?๘?๙?๚?ั๚?๛???ข????z??ะ????\????ล??็,~?N	ฟ'๖!ฝNภ>@๐พg}ท>ษ<ฝท๑พ	n?ยไ_ฟใ๗<ฦถi?ณืฟ2ฟE9?6B?5๙พVSzฟ(2พLโL?แnm? l>๑!ฟฎฟ๋ฮFฟชn0พh๒>ฌe?,,~?วJ?๔8ฮ>NฝVเํพGCFฟศsฟmฯฟWQoฟtIฟiฟฟ๕ตพํ$๕ฝ-ู=๙ฏ>h๛>0$?๏B?GZ?tฦk?๘,w??ข}?๛๕?ๆ~?? {?ย>u??ยm?me?2ง[?หฎQ?)pG?ศ=?ื2?ะภ(?Nํ?m?LK?W??z๖>Tฏๆ>ะปื>ทษ>1Pผ>?ฬฏ>?ค>#>ใฑ>ก>แ๖w>๕g>,DW>ฏH>Dร:>Xํ->๕!>Jฬ>ขe>nด>Y๓=ดโ=Y?า=Cฤ=๐ฌถ=็ช=>>=ฝF=-= =Qom=๗\=ฃM=d_?=?2=ฉผ%=_<=7==๘<UX็<}Iื<?Wศ<ฐoบ<~ญ<^sก<&><ะ<Q<&r<Va<ฌฑQ<ู"C<ป5<_๛(<@<=U<q,<hp?;๙ื๋;Bx?;;ฬ;ท๔=6Xฟฟฬ?Amฟำb???nฟ๊ื?๏ฌaฟฬน>UA๘>โฟ(๐ะ>D๗G?ช๏7ฟ$Q0ฟ'?
ก_??gVพ|ฟ์|ฟ{nฟ>ำy?8เJ?Aๅณ=ตG!ฟ้+|ฟr}aฟW&โพ@E๔=?A?๑Pj?ฌ?hฑb??๓!?J>ณฌฝgอตพ๖ฟ&ขOฟอIoฟฯ(~ฟ๛~ฟ๖:sฟ!_ฟDฟ6๑%ฟ3ฟณuวพEพ า
พ(?ผโฟฝ=~ัF>+ุ>หฝ>?kไ>]|?E??& ?%,?ช+7?h@?ภฑH?p?O?k"V?ี[?b`?"d?T*h?๖Pk?Dn?ซop?๗r?Lt?T?u?^5w?ax?ีey?Gz?.{?ลด{?ภG|?ว|?{5}?}?๑็}?ต/~?เm~?บฃ~?_า~?ฦ๚~?ฤ?<?SV?m?ฟ?อ???[ญ?nธ?ย?Tส?ั?ภื?%??ัแ??ๅ?]้?f์?๏?M๑?E๓?๚๔?t๖?ผ๗?ุ๘?อ๙?ข๚?Z๛?๚๛???๛??c??ฝ??
??M????บ??C#?LฟJ<?%ัพY>"ฆพซ?ิญcฟS{?Eๆฟ+ฯแพฝ??ฑ๗พฆ?jฟ{ฬ?>n?FCพฟฯ?พ?*F}?@เ>j๗?พ๐vฟส3`ฟชพ^]ช>ฺ+S?ฤ๙?\?=l?eYi=รlฟพ([6ฟ?lฟ?ลฟtuฟ๚Sฟ๋็"ฟ าพ3พ{]=ั>?๑ๆ>0จ?F<?	U?/@h??๑t?w|?	พ?ขk?<|?7ฮv?ฉo?h@g?ู๔]?T?ฯโI?ฬ??ีD5?!+?แ;!?ฏฆ?ญm?
?v[๚>Y\๊>ฃ5?>6ๅฬ>ซfฟ>๒ณฒ>หลฆ>%>p>ุC>?&|>ฌ๘j>ฑ๋Z>๖๎K>๒=>\ๅ0>ปน$>ๆ`>Mอ>๒>a๗=Wjๆ=คzึ=(ฃว=ัน=๒ฌ=C๗?=ขฯ=;m=ย=q=ึล`=n.Q=qซB=ฯ)5=ศ(=ฮไ=y=k฿=โ?<)U๋<?ฺ<๕หห<oฆฝ<8|ฐ<โ;ค<7ี<9<Y<วRv<9e<+OU<F<1ธ8<2ๅ+<๖<?<x
<็ <๋่๏;๐@฿;๛ภฯ;eEฟ1ฉฝ?q-?าชiฟฆLw?,rฟ.Q?F๊พะBพชg[?๑มeฟQYฝ฿w?bนหพษ๘fฟqน>FJ{?ฤvช=Zจfฟต-HฟL	>#f?\sf?>d#๗พ๙uqฟiqฟDทฟัbผ๕??฿\??:om?2ฌ3?S฿ล>๛N,=ถnพผฟozEฟg[iฟ	|ฟ?ฟิvฟLzdฟฒ=KฟKv-ฟฟฅ]ืพEีพ{(พ#ป7ฝบ=ฃ๑.>X?>+?ณ>ฮ?>J?>เ_?i?ใา)?๛ฤ4?e>?ุF?N?N?ธT??_Z?ยN_?Jc?Yg?hj?tpm?ๆo?็	r?%ๅs?คu?็v?
x?O+y?สz?3฿z?ง{?ท&|?}ช|?ญ}?}?Sี}?~?็_~??~?ไว~?ฒ๑~?็?D5?mP?๓g?S|?๘?@?{ช?๑ต??ฟ?vศ?่ฯ?Zึ?๏??ฤเ?๔ไ?่?ธ๋?p๎?ส๐?ิ๒?๔?๖?r๗?๘?๙?r๚?1๛?ึ๛?e??เ??L??จ??๙??>??{??ฏ??#พ#ถ'ฟ:~??pฟ*๙X?Z[ฟ?t??}ฟฯง:?
HผuKOฟ\b?hH,>ฟ๖ฟ?ฅ=Nเ?|?=๗mฟ(K)ฟh์ฯ>c๐~?ัว?ภxพ9dฟmrฟ,Q๗พ?<>๗V<?ว;}?R๓j?ซ?ฆ'>พ฿$ฟ๖ชbฟ~ฟข6zฟx]ฟ`/ฟkT๎พtๅkพฎฉ๋:9A_>า>W๓?ฤR5?เPP?Bud?Mxr?๙'{??T?ดฦ?ิ3}?้>x?#uq?ลNi?0`?mV?4IL?ฝ๙A?อฉ7?}z-?๓#?ั??i?ว?"7?>z๎>Pฌ?>'*ะ>{ย>[ต>?}ฉ>!>z>X>ู*>vแn>w^>ขWO>o A>??3>~'>B๕>ฤ4>/>?ฒ๛=ยM๊=มฺ=ห=๕ผ=(฿ฏ=7ฐฃ=xX=>ษ=ำ๔=ัu=d=AนT=u๗E=Y;8=แr+=9=ถz=ฟ,
=น =๚Q๏<~ต?<
@ฯ<,?ภ<?yณ<eง<Fl<'ข<พ<}z<i<ฉ์X<X?I<งู;<ฯ.<'ฌ"<๘`<?<ฯ<?๙๓;	ใ;iFำ;Vัsฟ้iA?Vb๘ฝฺฟฌพล?? ?ฟGป>4>ณ3/ฟh๕?T7ฟฆ"๏พฬY|?ะซผถ*ฟฏฟ?<๊~?pลผ>ส@ฟฉ๐iฟณCบฝ๕H?jWx?}๑็>๒๋ฃพ+'`ฟ{ฟใc-ฟค"พทPห>ึ}L?ณ|?ธอu?ืC?็๘ํ>/๚=}Xพ}f ฟ}:ฟ๋bฟSyฟๅ?ฟtืyฟ`yiฟTQฟท4ฟ๘หฟํ็พo?คพAFพถูฝ?<+=๘>#z>ีช>ฟา>4/๗>oุ?uH?ณ'?V2?&B<?{๘D?L?2IS?OY?7^?ฉงb?f?๎โi?ะl?8[o?โq?|s?u&u?sv?lูw?ฮ๏x?3แy?zฒz?ใg{?|?a|?s}?มi}?cย}?-~?ฒQ~?Q~?:ฝ~?v่~?่?V.?mJ?มb?ำw??เ?ง?iณ?ญฝ?ฦ?Cฮ?ํิ?ณฺ?ณ฿?ไ?ว็?๋?ึํ?E๐?a๒?4๔?ษ๕?'๗?W๘?^๙?A๚?๛?ฑ๛?E??ล??4????็??/??m??ฃ??Swฟโฑ[>X??2eฟ_{?โ|ฟ^$q?ไ>ฟ >@ฺ๗>ว}}ฟฤ*?C?'nฟฑพณๅs?ห<ฺ>\Hฟฐ๊Vฟุ&>ฐUr?วฏG?A#ฝJฟโ2}ฟทไฟXร?<ซฅ!?ฏ?u?๛นu?๊0?ฯ>ฯ:พภ๕ฟWฟปzฟฆ}ฟโeฟ;ฟ}กฟฑรพ
ไNฝ์W.>ตฃผ>๛	?.?พJ?ฮf`?ภo?ฮy?๒บ~?B๗?ต~?จy?5's?SGk?Zb?dถX?1ฃN?}YD?i:?อ/?pศ%?g?qง?ฃ	?๗?ฅช๑>วโ>lำ>bล>}ธ>ะ4ฌ>฿ฌ?>ใ?> ผ>ฏA>Oษr>{8b>ฑฟR>HND>5ิ6>B*>[>>๐l	>U฿?=๖0๎=ฒธ?=ะaฮ=?ภ=ฅหฒ=iฆ=?แ=4%=	'=zณy=Xbh=
DX=pCI=?L;=๔M.=?5"=๐๓=z=+บ=วN๓<zkโ<ดา<็ฤ<wถ<ๆฬฉ<U<1<๔ี<2ฌ~<?l<&\<:M<๛><ืธ1<3b%<ีๆ<7<F<ฯ
๘;Kาๆ;ึหึ;ๆ%พํ	z?ZญUฟAไ>ุ้Aพ}C>นใซพ}o+?ฤฝwฟM`?@พ2NฟUT?ศาป>ถuฟจพ_g?>Y?พฟ|ฟEฅพ๘0 ?์ห?V5?๓พธHฟเฟFฟ[ฝพ๑>^b9?๘v?{ท{?5PR?๖$
?CฒN>ๆพyJแพท.ฟฐ๖Zฟ๚guฟXฌฟ'C|ฟผ?mฟกWฟ{ฐ;ฟMฟ๋i๖พณพผบcพzนหฝะ<งำ?=g&d>"(?>ฏสษ>,๏>/G?w?ปC$?฿/?X:?บC?ห๐J?tิQ?ัูW?4]?Dฐa?ฎe?'i?&-l?ขอn?็q?Os?สษt?Hv?ฎw?Rณx?มฌy?z?{@{?๗โ{?ศo|?ฮ้|?S}?#ฏ}??}?@C~?อ~~?cฒ~?฿~?ฦ?K'?TD?y]?@s??q?ค?ึฐ?rป?ขฤ?ฬ?{ำ?rู???ใ?๗ๆ?R๊?:ํ?พ๏?์๑?ฯ๓?q๕??๖?๘?%๙?๚??๚?๛?%??ฉ??????ี????`????๒:=ฟไTi?ิพH็พ๔้?/(ฟึ?๗,lพR
พII[?มpฟิ5>ํY?ด"8ฟ?ฟL?ฟน/?bฟtฟ5ปฝฌ*X?u่e?ฏ(>(ฟ?โฟ{ษ=ฟrห๚ฝ
ค?acj?-น|?#ปD?4ฝ>ว*ชฝ.๛พจลJฟ?มuฟอ}ฟ๛-mฟFฟัฟฐญพ0Jาฝ	๚=8โฆ>ล ?ต&?$็D?\?nหl?Oฮw??๏}?B??ต~?Kรz?ฟt?โ)m?Hqd?๐Z?ค๐P?้ฎF?Z<?ฮ2?E(?_8?ธพ?1ค?โ๏?ฬK๕>?ๅ>6ฌึ>ศ>_ป>>๊ฎ>7ฃ>ม>>,๗>ํW>2ฐv>น?e> 'V>?{G>ห9>ั->0!>>ช>ฒ>๐๒=uWแ=ํภั=<ร=
ธต=ใ!ฉ=๖i==6Y=ส}=0l=ศฮ[=cL=X^>=)1=?$='m=^ว=?=K๗<s!ๆ<-(ึ<?Jว<"uน<fฌ<b?<9t<)<sl<เp<ข'`<ิP<B<ฉข4<>(<ฒl<<iu<ม?;๘๊;CQฺ;๗j,?๛ฃา>]ึuฟTQt?ษNฟU A?ภDVฟy?๛sฟภ?ฎ>๎{ฟe??8ู1?"ฺKฟ@กฟข):?ฃR?Fพ?ํ~ฟb$	ฟ%2แ>ฑ|?ฝธ@?๔ฝ๓<ฮ+ฟม~ฟm[ฟๅอพชO#>ผฮ#?๒หm?b?e๗^?pD? a>ษฝ;ฏภพไ4"ฟาRฟ์pฟฅ~ฟฝ~ฟrฟA]ฟ๙_Bฟำ#ฟยฟบยพดพนพ฿ผอ=;N>%#>hรภ>ฆ็>bฌ?N๋?!o!??_-?5ๅ7?P$A?@I?WZP?V?๛[?ต`?kำd?<ih?ซk?ั=n?๚p?ีคr?ขkt?Y๖u?ฮLw??ux?uwy?ฯVz?n{?Aภ{?ณQ|?ฝฯ|?๓<}?}?ํ}?4~?r~?_ง~?ี~??~?# ? >?X?n??๔?ก?9ฎ?.น?ฌย?ไส?า?,ุ???"โ?#ๆ?้?์?4๏?t๑?h๓?๕?๖?า๗?๋๘??๙?ฐ๚?f๛???????j??ร????R????K_+>qู_?oiฟK๖>๚
๔ฝ:eผ๏ฝSไน>fชCฟR๔?h,ฟ+?พ?}?@ฬพbqYฟผะ?โ็`?j_พ่ฟ3ษญพJๆ1?ก๚x?๎น>กี ฟzฟฯ~Wฟๆพ฿ล>?Z?]ี?KV?๏>Nฝ<ัพ;<ฟ^0oฟS๛ฟ้Qsฟ`)Pฟฟํวพ|Gพห=ษ>ฯง๎>ณื?vฬ>?,W?ฑi?โฮu?M๔|?ตุ?ซ>?ฌึ{?m<v?F๖n?vf?o]?g1S?ไ๙H?ฆ>?{]4?]>*?ซ_ ?1า?ฦก?Jึ?฿่๘>ๅ?่>>ู้>ซห>M?พ>Aฑ>มฅ>ฒ>x1>m>z>.i>๎Y>uจJ>oม<>0ษ/>มฐ#>ใi>็>>ฑ๖๕=	๖ไ=ใี=>`ฦ=Wคธ=ฺซ=๒=๚?=X=M๐=ฆ?o={Y_=M?O=อoA=4=_'=Zๆ=ฉ==VH๛<iื้<;ู<Wส<รrผ<ๅ]ฏ<n1ฃ<A?<]R<ฬ<รt<ลc<๕S<>E<z7<Iฮ*<๒<้<ถค	<Y <ฅc๎;ฐึ?;ธc|?$j๘พxรัพ%X`?-~ฟ5๔?๓]ฟ๑n?$%ฟฅ<ื=?ูgvฟข>eฟj?
ฟรสUฟn๔>ฃฤs?ฃิฺผQอpฟ8ฟฑ,n>%n?๗7]?pS>ำ1
ฟBvฟlฟr2ฟ?น=?Ab?ผ๖?ฆฐi?8-?ฅzถ>ีฑC<แ#พ6ฟ.`Iฟญkฟb์|ฟNฟuฟ๘|bฟฺยHฟ<ฎ*ฟv)
ฟrัพพ~|พัฝๅ1=^โ7>๓>ชท>{์?>H?)ฏ?
?ู*?ัซ5?N0??HG?ไฺN?@U?ฌึZ?<ถ_?l๕c?
จg?ฅ฿j?ลซm?p?จ6r??t?[ฃu?ฯw?i7x?OAy??'z?ผ๏z?๛{?"3|?@ต|? &}?ฐ}?R?}?ค%~?(e~?-~?ิห~?๕~???ำ7?ฅR?฿i??}?i??ซ?แถ?ญภ?*ษ?ะ?เึ?c??)แ?Kๅ?เ่?๙๋?จ๎?๛๐??๒?ฝ๔??๖?๗?ฐ๘?ซ๙?๚?@๛?ใ๛?p??๊??T??ฐ?????D????ิk?Is>2gฟ=w?$Cฟyพ ?ขฟ(ฟฦT?ต}ฟอ`?ทX}พ~L*ฟฌซu?(ูผณZ{ฟv>@ศ{?฿ๆผ๛8xฟิษฟ?ฌ?๙?'z?อ๑จพx8mฟโPkฟฅษีพตg>๘F?K?e?รW?Yฑ๎=ุตคพ+ฯ,ฟwgฟฟกFxฟdYฟ๏)ฟฅโพ๑๚Rพ_ศฬ<mวt>฿X?>ุ?ฆp8??ฒR?(,f?๘s?ศ{???ฃ?จส|?>w?Sฌp?hh?I5_?YeU?O:K? ้@?6?ฅp,?8"?อแ?8?$บ?ฯ?>sf์>#?>aทฮ>ยม>ีPด>xIจ>ณ?>k>>{~>ึ%m>๔\>ฦิM>mท?>92>D&>~ะ>ๆ#>V1>6ู๙=m่=ด~ุ=ภษ=ป=Aฎ=4{ข=ส8=pฝ=๛=นฬs=#ไb=/'S=;D=฿6=ธ.*=_=๑a=r'	=E?<\ํ<G?<ธอ<apฟ<b&ฒ<zศฅ<HF<<%<ขฅx<bg<NRW<y_H<Kv:<T-<kx!<B<ิ<า<R,๒;\แ;กศ>ท}ฟS??>Gอ>m4%ฟ๐=G?q@ฟ4z?`Wพs๗พ
x?ธ#?ฟผ๘พx๕?Y2BพฆxฟM9>๗?<}z> Sฟบ\ฟKf<฿ชV?mฉq?{ภ>ษพlhฟัwฟโ!ฟHดฝrูไ>T?^A~?8dr?u฿<?e?>ัฑ=pคyพบ0ฟWv?ฟaฏeฟๅzฟ๋ฟxฟ=QgฟึNฟa1ฟgฟูเพณnพฐ!9พทvฝ0wY=!>ข้>ฅฎ>Eฑึ>Fถ๚><j?ฌ?ZJ(?Dk3?ฦ5=?'ฮE?&VM?ผ์S?ฎY?ขณ^?&c?๔ใf?5j?m?Go?ศฦq?เชs?Ou?ฐปv??๗w?O
y?-๘y?fฦz?'y{?|?X|?ฐ}?}s}?ำส}?{~?X~?อ~?๚ม~?์~?x?l1?M?e?ิy?ฮ?`??จ?ด?งพ?iว??ฮ?ี?@??-เ?qไ?"่?U๋?๎?๐?๒?a๔?๏๕?I๗?t๘?w๙?W๚?๛?ม๛?S??ั??>????๏??6??s??ซS?ๆ7ฟ	พร[?#ณฟF1{?|ฯzฟ"??PXiฟ4K?bp>้สlฟ5ฏB?"Jป>-|ฟ}รฦฝฐ}??>ู	^ฟUAฟถb>Az?้Q4?#พtXฟตxฟ๚Nฟท9แ=$ษ/?6Cz?ะฉp?โC%?jH\>ๆmพำฟฃr]ฟบฆ|ฟ๗|ฟตaฟK05ฟป๛พCพยผ(|G>ฆว> ?โี1?หขM?ดb?-q?wkz??ฌใ? }?U็x?เKr?QGj?|@a?VW?
pM?#C?hั8?
.?๘$?ํ?{?i?F ?ฬ๏> [เ>๕ภั>j๚ร>๒ท>ซะช>ภ^>ษฃ>?>|/>ฐศp>Y`> Q>?ฌB>๊N5>ื(>แ6>`>๑F>ป?=ก2์=^??=#งฬ=ฆ|พ=าKฑ=บฅ==}๏=บ=ฝw=ภnf=sV=ขG=
บ9=ื,=ดุ =6ฏ=ฺK=๋?=LC๑<Oเ<ฟ๎ะ<?mย<฿๎ด<_จ<Nฏ<รฮ<}ฏ<%|< k<ฏZ<ํK<`=<^:0<G?#<<O<J'<?๔๕;แไ;๚?ฟู๋2ฟฎTw?ๅ`ฟRF=ฉE>Q#MพZNผะา>?*[ฟกุx?>ยพ<&ฟ๎An?ฃง1>ส~ฟีC	พม(v?'ี?>+ฮ'ฟ\uฟฅRพMท5?N}?ฅฏ?ฌฌrพgwUฟ~r~ฟ:ฟWWพ|ฎ>C?ฆz?ำ?x?rK?ฎp ??%>ส3พฑ๑พู4ฟอ๗^ฟีgwฟํฟg{ฟ7ผkฟะTฟํ%8ฟฆzฟ<๎พ๓ฒซพXฃTพxฏฝึ๐<งB>no>?Dฅ>ฎdฮ>gJ๓>น
? ฟ?๛ณ%?ฅ#1?ศ4;?-D?&ฬK?FR?VX?Uญ]?/b??f? i?l?o?6Uq?GHs?๙t?qqv?ทw?vาx?มวy?kz?ลT{?๔{?|?๗|?๙^}?น}?~?ฐJ~??~?๘ท~?่ใ~?๖	?์*?xG?1`?u?&?5?ฆ?*ฒ?ผ?กล?sอ?:ิ?ฺ?,฿?ใ?b็?ฏ๊?ํ?๐?(๒?๔?๕?๗?7๘?B๙?)๚?๒๚?๛?5??ธ??(???????'??g??ฮูผ?|ฟ/?ใ๔`>ฝฉ2ฟฯ3`?{ลfฟํJQ?:Fฟ๙ข<ฝ)?จ|ฟ -?>`ค1?
ข[ฟ?ณุพKpf?	ฌ
?ภN3ฟชceฟr์h= ?h??U?ช<=๙?<ฟดMฟ๚`,ฟลฝ{?ฯชq?ูy?้G9?Y>QRพญ	ฟ฿cRฟeูxฟ>~ฟiฟซว?ฟ
ฟ?@พฅ!ฝ
ศ>oณ>ถ)?j?*?-VH?Eก^?ฎn?ฯ?x?l~?๐??๛S~?z?วิs?l?ๅ;c?=ฆY?๙O?TTE?| ;?yร0?ฺธ&?:๕??z
?ำ?L/๓>ไใ>Fศิ><ีฦ>ฑน>ชVญ>ึผก>ฦ?>ฦช>๐ >ทjt>qพc>ะ+T>ขE>B8>อi+>	>๛>f\
>วฮ >ฅะ๏=เ;฿=fสฯ=งhม=Mด=0ง=A๐=!=ไ=ดh{=R๙i=ืพY=คJ=<=\/=?Q#=x?=?p=H=8๙๔<V๘ใ<p%ิ<kล<Yทท<๖ช<S<๖<ีล<S5<n<ฦ^<`ขN<์I@<i๐2<#&<ฅ๔<2<ย/<ซฝ๙;๖f่;ๅ่ฟอ>UR?ฟyฟiX7?Q#๗พฅข?>skฟ1}U?/๓ฟ ฬ??ฎ=ข[gฟฐU8?๚?๐gฟ฿พ?0W?ท6?JMใพ๓ฟ+ิพ้ ?๋บ?โั,?ก๗ฝD=ฟ?ฟWำOฟโ้จพ?k>ฎง0?qBs?ืq}?รำW?gู?ํUp>'ฺฝ6หำพศ)ฟลWฟ_?sฟYTฟร}ฟไปoฟ:Zฟ>ฟฎ`ฟย?พ#ันพ/?oพ๙ใฝ?pบ;นฎ้=๐Z>ฃ๚>bฦ>{อ๋>ิฦ?]ษ?#?
ี.?e-9?hDB?๎<J?%7Q?lPW?Wฃ\?ฮGa?(Se?dุh?T่k?ำn?๔แp?3ไr?ฃขt?&v?5vw?รx?y?อqz?ำ/{?ิ{?Fc|?๗?|?%J}?ง}?s๗}?%=~?y~?ฯญ~??~?V?Q$?ภA?=[?Pq?o???Uฃ?ภฏ?บ?ัร?โห??า?ุ๋?(??ฑโ?ๆ?๊?๗์?๏?บ๑?ค๓?L๕?ป๖?๙๗?๙?๛๙?ส๚?|๛???????v??อ????Z??[[ฟ๊๎พน~?9ฟแธผูศธ>๎้พ๎ฐ>93ถผ๐๗พFo?ช^ฟ,Mฉ<ขj??.ฟw3ฟิU8? C?วm๖พtกzฟ7พKL?+เm?ฬop>zPฟฆ์~ฟ^Gฟ_ด3พ2๑>ํZe?d@~?ป(K?}Hฯ>จ๙Eฝ?์พ๗Eฟ?ฅsฟTำฟ~soฟัซIฟฟ;ถพแb๗ฝ.ื=9>๛>์#?XฮB?ุZ?mตk?a"w?ู}?[๕?"้~?ล&{?โFu?ฦฬm?a'e?๎ฒ[??บQ?|G?1(=?฿ใ2?ฮฬ(?๐๘?Hx?V??~๖>ัมๆ>Jอื>3ฎษ>ท_ผ>s?ฏ>๐ค>๙>่ฝ>เ>้x>"g>VW>ัH>@ำ:>A?->๖">?ู>ตq>ฏฟ>vn๓=:โ=ํา=Tฤ=ดผถ=ช=้K=vS==6=ุm=
]=YตM=๑o?=ฆ'2= ห%=ทI=ข=ฃ=!ฏ๘<Yl็<\ื<3iศ<ำบ<ญ<Wก<'K<,?<&<;r<ja<ารQ<ผ3C<sฆ5<?	)<ซM<็a<;8<X?;c์๋;ฟงb?)ฬ=UฟO๏??พnฟลฮc?:pฟส๏?@:`ฟ?Pด>Mึ?>๒ฟถฦฬ>ใGI?6ฟ;ค1ฟท%?พe`?ๅPพ[ฺ{ฟฟฆ8ฝ>ีืx?มK?ป=๚ ฟ๎|ฟแฺaฟpใพั๓๎=ร?j?gณ?ํb?ฬO"?>ใ4ฝ,ตพฏฟJqOฟ(.oฟ ~ฟ~ฟsNsฟณ_ฟFขDฟ&ฟ[ฟ"ฦวพzพ฿gพ#ฅผปผ=่YF>ิก>ฝ>@ไ>ภh?ูห?บp ?,?ฑ7?ไv@?จH?`ีO?_V?ฏ[?ย\`?wd?E&h?nMk?2n? mp?ฆ~r?}Jt?ูu??3w?7`x?ฑdy?Fz?S
{?ด{?G|?ฦ|??4}?ฎ}?็}?e/~?m~?~ฃ~?+า~?๚~??๑;?6V?๖l?ฉ?บ???Mญ?bธ?๛ม?Kส?}ั?นื???ฬแ?ุๅ?Z้?c์?๏?J๑?C๓?๘๔?s๖?บ๗?ึ๘?ฬ๙?ก๚?Y๛?๙๛???๛??b??ผ??
??M???eฟแV?>z8?ึ{ฟ5*?:ฅพ*๒2>ณว~พ๑ฃ?e[ฟ?๏}?หฟ๖?วพ๔?-ฟพfฟ%็๏>vอj?มgพ6ฏฟ?ฯพ4X%?+|?ฦิ>้ป้พwฟDค]ฟz๊?พwุฒ>	U?๎??ฑZ?m??>7=ุฤพGB8ฟ?mฟ??ฟาtฟ}ำRฟo!ฟฎKฯพ๛-พv=(>ศO้>ศข?์=?|2V?ํชh?6u?oฅ|?ํฦ?^?โ|?ขv?ณro?ฬg?Iฒ]?๛ฯS?นI?qH??*?4?ล?*?๘ ?ปe?W/?~]?$๊๙>?๐้>๚ฯฺ>Gฬ>Rฟ> _ฒ>vฆ>]I>dะ>I>Bฌ{>j>ฎZ>K>แ=>j0>จh$>R>?>wฐ>๗=l๘ๅ=ึ=[@ว=uน=้ฌ=ง?=c=(=;=Rq=YV`=ชฦP=?JB=์ฯ4=D(=๒=น=๛=e?<Zเ๊<หฺ<หfห<KHฝ<$ฐ<Z๊ฃ<X<๒<ำ<}ุu<<วd<DๅT<F<|\8<ฺ+<ฐฆ<3<ณ@
<ง <ฯq๏;mZแ>ฉืf?~1ฟ๔๎7พ8??ทMrฟ|?๓wฟ|[?ก}ฟลฤพQS?`ฐkฟt<wt?เพ(bฟ`?ห>y[y?QK=jฟๅpCฟืw0>ๅh?ฤc?๛:>๗ ฟsฟย฿oฟโ;ฟN่ฟ:b?ี^?พ?โRl?ธ1??.ม>๕=ฒฃพเ7ฟ๋ฎFฟOjฟฆQ|ฟฟCrvฟ:เcฟฯ|Jฟl,ฟร2ฟุีพ9พ5%พ่*ฝลฏ=Uฎ1>P;>_ต>ข?>ฒ ?ฦ?ฤ?=#*?ผ5?ฐฃ>?G? oN?7โT?aZ?|n_?ํถc?คqg?Uฐj?ฅm?^๖o??r?๑s?๚u?๐v?า%x?2y?ฃz?Eไz?{?*|?สญ|?}?}?xื}?p!~?a~?~?ษ~?พ๒~?ฯ?6?Q?h?ี|?i?ก?ะช?;ถ?ภ?ญศ?ะ?ึ???ใเ?ๅ?ซ่?ฬ๋?๎?ู๐?แ๒?ฃ๔?)๖?{๗?๘?๙?x๚?5๛?ฺ๛?h??ใ??N??ซ??๛??@??M๋ฝ๗<{?SึzฝOฟCแ~?า?Zฟๆ<?ฆKAฟnd??๑ฟz<Q?ฏ?พฆ;ฟภ\n?จ=PT~ฟBฐ.>K~?.w:=b9tฟชุฟoํ>??M?ซwพอxiฟซnฟFฎไพ๑b>dB?V~?vดg?ก?M>ู:พ?Z)ฟ?,eฟง~ฟ`)yฟ"6[ฟX,ฟุ็พ๓]พ2yr<kTk>@ื>#?7?KจQ?ูle?s?{?ญs?ด?ฦ๙|?0ๆw?=q?ฮh?.ค_?ืูU?ณฑK?'aA?G7?ฏๅ,?๔"?ำO?็?+?0B?>ํ>Nะ?>rZฯ>aทม>Mแด>#ัจ>๑~>8โ>+๒>ฟK>ไ่m>Hช]>?~N>&V@>H 3>ฮ&>5Q>?> ก>ฉ๚=tV้=k3ู=,ส=B-ผ=,%ฏ=ฃ=Cท=,3=!i=มt=ขc=๒ืS=ม%E=,x7=;ฝ*=+ไ=a?=R	=u =XT๎<uษ?<adฮ<มภ<กปฒ<]Sฆ<ว<ุ<	<๕uy<u$h<ถX<[I<;<ถ.<ต?!<ภ<*I<ุ<;๗๒;ีN~?9D>?ฟชี?m^ฟ=ฟ4ด-?ู'ฟ ็>ไ^ฉผ๕ฟ3}?Q#.ฟmมบพฎx?้ฝU?|ฟฮ์=ผ?>Kฟฒไbฟ๎ฏฝxP?Bูt?า>/นพไeฟBขyฟn&ฟศ?่ฝaดู>\ำP?}?^๓s?๏๙???)ไ>ืั=wkพ8ฟภK=ฟfVdฟ	๊yฟำ๘ฟๆ%yฟ๎Fhฟ;Pฟ9๐2ฟๆฟ8(ใพผn?พิ็>พ๕๒ฝ=%E=๏>๐>ฌ>ฆ๕ิ>พ)๙>ิน?C?<ภ'?๑2?ูส<?apE?M?๘คS?poY??|^?ไb?บf?
j?+๘l?~o?"ฏq?Xs?A=u?:ฌv?๊w?ฎ?x?๎y?จฝz?q{?|?จ|?ย	}?8o}?ว}?G~??U~?e~?ๅฟ~?ล๊~?่?0?ํK?d?๓x??ธ?Jจ?ด?9พ?
ว?ฌฮ?Hี???๗฿?Bไ?๚็?2๋??ํ?f๐?~๒?M๔?฿๕?:๗?g๘?l๙?N๚?๛?บ๛?M??ฬ??:????๋??3??ณF?K=?ทสLฟOพะJ?แ?|ฟ?ฯ~?xn~ฟล?~?จV`ฟtศ็>ๅำ>h|rฟ8?G.ี>yฟฉภพ๐๋{?้ม?>t?XฟG2Gฟ>:ะx?sx9?q๒๑ฝ?Tฟ?zฟ๚เฟUธ=G,?วIy?r?F0(?๕Cj>ฆๆ`พZฟ๘๛[ฟ44|ฟTs|ฟ่หbฟnถ6ฟภ?พจ,พ๒๙ผhA>l?ฤ>\q?4ไ0?o่L?้๛a?๖ัp?ฮ6z?ง๛~?ถ้?\บ}?#y?Dr?์j?a?uุW?hพM?=rC?% 9?|๊.?s๋$?6?ดู???LK ?/F๐><ฮเ>ฌ-า>฿`ฤ>Vbท>5+ซ>ฐณ>`๓>แ>.u>๛Jq>Pำ`>4rQ>C>?ฑ5>U3)>ๆ>ถฐ>จ>บF?=Sด์=*V?=ฃอ=hๅพ=\ญฑ=^ฅ=้=2>=?O=$#x=ถํf=3้V= H=h :=R6-=`1!=ผ=ง=d่=Tศ๑< แ<๕aั<6ูย<ฆRต<^ผจ<ท<-<R๚<k}<ฎk<'([<*๑K<ศ=<0<บX$<ห๏<ขQ<-p<ง|๖;5!"?ึ,ฟaฟg}?76ฟ>Mล=\โฝ|2ภฝตค๖>๛Cdฟ/,t?0'คพzo1ฟมh?.๙d>3R}ฟ6พีs?์ี?ํฬ ฟkWwฟฆํpพธt0?5~??[พ^Rฟ๖~ฟ#[=ฟV๑hพฟฆ>ฝ?@?๑5y?ำฟy?b?L?W๎?ฐ/>ีํ)พ6{ํพ`O3ฟs๘]ฟผ๊vฟ}แฟ!h{ฟRlฟ?`Uฟ!9ฟปtฟ<๐พtฏญพWzXพ\^ถฝธคี<ใ>[l>น๘ฃ>ฬ9อ>๗>๒>คฅ	?aU?V%?Zั0?m์:?ณฬC?K?ฉcR?โVX?M]?Vb?ๅ f?oi?ลkl?o?,Eq?Z:s?iํt?๔fv?}ฎw?สx?๊ภy?}z?กO{?๐{?({|?ช๓|?\}?ถ}?้~?อH~?~?ถ~?ฎโ~?ๆ?*?ฌF?_?u?ข?ร?ปฅ?ิฑ?Mผ?`ล?;อ?	ิ?๎ู?฿?sใ?G็?๊?vํ?๒๏?๒?๖๓?๕?๙๖?.๘?;๙?#๚?์๚?๛?1??ด??%???????%??Qxs?ฐฦฝยฑyฟถ1?ลฦ'>Uต(ฟ๓ืY?๎raฟ6ะJ?ฐฟฑผ)0?}ึ~ฟGMอ>hฎ6?:+XฟCลใพ2ไc??นฝ/ฟฎegฟvj$=ตMg?W?6s=ญ:ฟฑฟ<m.ฟม?0ฝm?๚๊p?าy?ท:?ำ๐ข>จ	พ{ZฟQฟxฟูฌ~ฟบiฟx@ฟr๘
ฟพ'ฝ>น*ฒ>้?*?๔G?โX^? Zn??ภx?ํ^~?y??_~?อ(z?ฉ๏s?b3l? _c?บหY?ฟมO?{E?ฐ';?๊0??&?ว?ดช?ย
?ง๓?]l๓>พษใ>๏?ิ>ฦว>โน>=ญ>็ก>ฺ>Mะ>D>\ฌt>ล๛c>eT>ืE> C8>N+>eศ>fล>
>฿๑ >๐=วx฿=ะ=xม={5ด=๚นง=โ=.I=ี6=zญ{=V9j=l๚Y=w?J=ศ<=fฏ/=~#=&=๙=Rร=L<๕<ย6ไ<_ิ<ชกล<ฉ้ท<_%ซ<ๅC<5<๋<pX<็?n<I^<๘ฺN<~@<k!3<ฟฑ&<<Z<T<๚;:8พฺส~ฟเa>ฟEI?]|ฟx@?lvฟ8๒>ฝ6ฟแํZ?V๖ฟยฺ9?J์ย=ฉjฟW3?#	?่EeฟE@้พถBT?แ%:??พ.หฟยg?พC
?^?-/?}.~ฝSa;ฟสยฟS9Qฟy$ญพnบc>s>/?ทฐr?"ญ}?@ชX?I?ou>ขะฝoงัพโม(ฟ๎?VฟUsฟฉCฟํ7}ฟ pฟ0fZฟU๕>ฟก?ฟ๎ภ?พิฮบพ?่qพฏๆฝS;?ๆ=ฉ|Y>#R>oล>|E๋>8??zๆ"?ซ.?{9?$B? J?QQ?ป:W?k\?P7a?ฬDe?ๅหh?v?k?an?ภูp??r?tt?ต v?qw?บx?y?ฤnz?2-{?@า{?La|?A?|?ฉH}?ธฅ}?V๖}?.<~?ฎx~?ญ~?zฺ~?ห?ู#?WA?ใZ?q?+?ร?"ฃ?ฏ?[บ?ฐร?ลห?ลา?ีุ???กโ?ๆ?๙้?ํ์?{๏?ฒ๑?๓?F๕?ถ๖?๕๗?	๙?๘๙?ว๚?z๛???????u??ฬ????#๕>+[ฟณํพ:ภ~?รmฟlฑผcธ>:้พXฐ>ฦปฐผ<S๗พฆo??^ฟณ3ฅ<Dฎj?ฆฟู3ฟมD8?ี.C?oH๖พ}ฅzฟ(ฺ7พ๎L?ๆm?ฉp>{Eฟx๋~ฟfGฟDเ3พๅ๘๐>ฃVe?pA~?ั-K?ทVฯ>๙Eฝญ๐์พ๓Eฟ3คsฟำฟGuoฟถฎIฟฟึถพ๖๗ฝ#jื=ฬ2>ภ?๚>e๊#??ฬB?Z?ดk?ำ!w?}?R๕?J้~?'{?OGu?Iอm?๔'e?ณ[?กปQ?7}G?ุ(=?ไ2?oอ(?๙??x?V?"?๖>ษยๆ>5ฮื>ฏษ>`ผ>7?ฏ>งค>ฅ>พ>w>x>ค#g>|WW>ถH>ิ:>	?->ฑ">ํู>Wr>Fภ>o๓=@โ=}๎า=qUฤ=ฝถ=Zช=?L=!T=ค=ฤ7=๋m=]=HถM=ะp?=u(2=ภห%=jJ=I=>=Bฐ๘<fm็<]ื<jศ<ซบ<^ญ<ก<ีK<ฮ?<+'<<r<ka<ฦฤQ<4C<Eง5<ร
)<aN<b<ู8<~?;ผwฟ?Iฟฃศb?Iส=cUฟ๐?าnฟฬๅc?ฟJpฟฟ๐?&`ฟ๒	ด>?>ฌ๒ฟภฬ>oYI?๙|6ฟ๐ต1ฟ*ฆ%??o`?DRPพื{ฟ7ฟ๖ฝ>Wิx?%K?๐ป=ฆ ฟ๙|ฟร฿aฟ?ใพqฌ๎=้ผ??j?มณ?๐b?ฅT"?)>?ฝธตพฯซฟบnOฟต,oฟฉ~ฟw~ฟxOsฟ3!_ฟคDฟ&ฟ0]ฟYสวพพบoพ(เผVญผ=ฃSF>๛>{ฝ>ธ=ไ>นg?๏ส?๋o ?ำ~,?7?Wv@?จH?๔ิO?V?][?z\`?8d?&h??Mk?	n??lp?~r?bJt?~ูu?ว3w?%`x?ขdy?}Fz?H
{??ณ{?G|?ฦ|?๙4}?ฉ}?็}?`/~?m~?zฃ~?(า~?๚~??๐;?4V?๔l?จ?น???Lญ?aธ?๚ม?Jส?}ั?นื???หแ?ุๅ?Y้?c์?๏?J๑?C๓?๘๔?s๖?บ๗?ึ๘?ฬ๙?ก๚?Y๛?๙๛???๛??b??ผ??
??ํ@-ฟmฟ0ฟ>EฃB?ํxฟ4 ?๘พ๋>ํฌUพl=๖>ฏ-Vฟฌ฿~?Y@ฟจ8บพ3ป?Uโฃพccฟo๙>ตh?Zzพโ?ฟ ๒วพัD(?้{?2wฮ>ฃ>๏พยLxฟH\ฟธ'พ04ท>@ดV?$??7ถY?K๑๙>}=ดขวพม;9ฟฝmฟ"ๆฟ}tฟน9Rฟกซ ฟ\อพ!ฌ)พ:]=?>๊>ึ#?Ds=?ฯV?ณแh?Yu?นท|?Dห?W?โ|?v?Vo??โf?ั]?๗ซS?๒vI?#???ู4?๘ท*?ฬี ?*D?!?น>?ฏ๙>Wน้>vฺ>นSฬ>ฆ?พ> 3ฒ>ูLฆ>ฝ">6ฌ>nเ>๏l{>์Jj>jIZ>yWK>ฝd=>a0>ษ>$>J๎>|b>>ํฬ๖=ฝๅ=ฟูี=Tว=Eน=ฌpฌ=R~?=
_=k=a=wะp=ฦ`=P=๛B=ก4=๋(=ฝn==(y=5$?<ค๊<งZฺ<2ห<ฌฝ<]๗ฏ<?ภฃ<)b<ฮ<ๅ๕<Vu<vd<ฎT<ง๊E<-8<ศc+<ฌ}<k<.
<u < w<ฟโzม>xfm?J&ฟ?oพEญG?%๖uฟห}?U]zฟs`?:ฟฮๅฟฝfN?wnฟฃ;=r?J6๋พz_ฟ?bี>*;x?&=ฌkฟิํ@ฟ฿>>=Cj?-Ub?ึBy>gmฟอsฟๆoฟทkฟ<ญ?|l_?อฯ?๎ผk?mด0?.มพ>ศ#ๆ<หฮพ ฟ#MGฟฎrjฟv|ฟ }ฟ๖>vฟcฟนJฟ-,ฟYผฟิพ$พ?s#พI$ฝOร=3>๔฿>ฒต>(?>R> ??๛?	๓?ฑL*?<05?ภร>?+G?N?ท๗T?%Z?ู~_?-ลc?~g?ปj?m??o?ดr?4๗s?Ou?'๕v?ี)x?5y?งz?โๆz?P{?|,|?~ฏ|?!}?]}?ุ}?f"~?Xb~?ฝ~?นษ~?H๓~?G?t6?uQ?ุh?}?ฃ?ิ??ช?`ถ?>ภ?สศ?0ะ?ึ?%??๓เ?ๅ?ท่?ึ๋?๎?แ๐?่๒?ฉ๔?.๖?๗?ฃ๘??๙?z๚?8๛??๛?j??ๅ??P??ฌ?????2|ฟpพuๆr?๖.m=โ^ฟศ{?ถDMฟส+?๊ญ2ฟRฯZ?ๆ~ฟฒฅZ?JูUพ:1ฟs?
`e<ฏ|ฟkoZ>ฒใ|?s <ฤvฟV"ฟษ๚>??I?๒hกพชอkฟ?ฆlฟ{?พ\<u>ท5E?อฤ~?ผf?1?ะ;?=		กพiy+ฟ<Wfฟไ~ฟQ?xฟZฟ
เ*ฟ4ไพ>WพฝUฌ<q>3ยู>ิ.?j้7?KR?(โe?hs?~ญ{?R?)ช??|?๐บw?๚ฮp?ฤh?p`_?ฉU?ผhK?ทA?dษ6?",?zฎ"??Iล?eแ?ฬ?>_ญ์>ฌf?>บ๖ฮ>pYม>๔ด>+~จ> 1>Q>๎ญ>ฬ~>qm>?:]>ืN>๕?>ผล2>ญy&>}>R>ย\>*๚=ส฿่=ๅฤุ=ลษ=jอป=๎หฎ=๘ฏข=๊i=+๋=&=๘t=ๆ-c=ืkS="มD=7=f*==โ=T	=$?<ฆฺํ<4X?<๛๚อ<ฌฎฟ<Z`ฒ<k?ฅ<{x<Hฟ<ฤ<๖x<ๅญg<`W<ฎ?H<๙ฒ:<ฬผ-<๗ฌ!<s<<*I<ฌึ/>?w?Dฏก>ฟซ?๛>=H>6๙ฟฌฬ=?WT7ฟeโ?จฝฝุ&ฟ๗Zz?yป8ฟนฝ?พ๙?n>$พzฟ๕$>???๕E>Pฟ?/_ฟ~ปซNT?&๒r?Sว>XCรพ้Agฟ?xฟ;#ฟณศฝFเ>)ุR?~?s?๑>?k฿>ูBพ=??sพ6
ฟ?>ฟZ*eฟำGzฟ๑ฟcอxฟ7ฑgฟQOฟv2ฟ?ฟ?Kแพ2พ?_;พง~ฝ฿Q=รห>ฤ>ภญ>ึ>Y๚>ร%?ํo?ฟ(?<3?G=?รฉE?J6M?ใะS?วY?n^?ญc?฿ำf?'j?Xm?ถo?ฝq?้ขs?(Hu?ฐตv?ษ๒w?ฬy?D๔y?รz?7v{?|?#|?ฦ}?ีq}?cษ}?=~?๒V~??~?+ม~?เ๋~???ๆ0?ฅL?ญd?}y???คจ?Yด?|พ?Dว??ฮ?tี?(??เ?_ไ?่?H๋?๎?v๐?๒?Y๔?้๕?C๗?o๘?s๙?S๚?๛?ฟ๛?Q??ฯ??=????๎??Qฦพฌ ?ซ๔Z?B.ฟEพT`?b?ฟD5y?๋่xฟs๏?n\lฟผH
?คฦV>.gjฟ$TF?wfฑ>๎์|ฟ8?ขฝู9~?๒>pฺ_ฟ๛?ฟถ~>0&{?X2?Tพ็นYฟD+xฟฟ๐=ฑ1?z?ฤ p?ๅ'$?W>วrพUภฟณ?]ฟ๕ฯ|ฟ;?{ฟ๓Jaฟฏ4ฟI>๚พ๗/พแญผฦแI>ฒศ>m?{02?่M?}ถb?Oq?z??7แ?น}?ธึx?6r?.j?Q%a?ฃoW?RM?LC?มณ8?แ.?$?า?y? ?]ๆ?>ู๏>ะ/เ>ั>?ำร>ฏ?ถ>ฎช>อ>>ุ>๘z>D>ญp>ะ+`>ะีP>B>ณ)5>[ด(>>_B>ึ*>"?=ู์=๎ฏ?=า|ฬ=>Uพ=!'ฑ=แค=ฟt=โั=+๋=ogw=?>f=FV=CiG=9=6ณ,=Zท =,=๗.=	=B๑<พUเ<iระ<ชEย<Wษด<<จ<อ<ฐ<W<ยS|<Sฯj<-Z<ตVK<ำ8=<ฯ0<B?#<๗{<ูๅ<เ<ไ๗k?โfG?ำคฟญ;ฟดt??๋๕พฬh<Oj>-Toพธ<ฒชฤ>pWฟDNz?วำอพ?!ฟi$p?q9>2ฟษฏ๐ฝ!8w?f๘>ฃb*ฟ8tฟฐeFพ่ง7?฿๋|?ง?DV{พ
Vฟ<:~ฟั?8ฟQพ/zฑ>รpD?Kz?mณx?๑eJ?{??>ท!>[~7พก#๓พ๛l5ฟ๏V_ฟwฟ^๑ฟพ๙zฟMkฟ:MTฟDฮ7ฟZฟ๗ฬํพ|๓ชพd1Sพ<Yฌฝ๛<p>Bp>ธมฅ>ีฮ>้ฎ๓>yI
?ฐๆ?ื%?B1?๖O;?$D?แK?ฆR?HX?=ป]?น;b?'f?.i?l?to?=[q?Ms?	?t?auv?ปw?lีx?Sสy?ฆz?ณV{?9๖{?x|?D๘|?`}??น}?็~?eK~??~?ธ~?^ไ~?\
?D+?ฤG?s`?ิu?X?`?Cฆ?Jฒ?ดผ?นล?อ?Lิ?'ฺ?:฿?ใ?l็?ท๊?ํ?
๐?.๒?๔?ฃ๕?๗?:๘?E๙?,๚?๔๚?ก๛?7??น??)????฿??n์?ั?๛L#>y๖ฟุ?>๑บ>sJฟN[n?{rฟS`???"ฟ๐=Jq?'๑ฟ.ไ?>้ซ#?>dฟฺบพ๙l?nJ?>Ni<ฟkน_ฟaูฮ='(m?วP?-ย'<?~BฟE~ฟิ?&ฟะปห?s?ษw?ณ5?ๆe>ร้!พณ%ฟTฟชyฟฦ*~ฟ๘ฤgฟพฺ=ฟNุฟ3|พญฝrW">n]ท>ฝม?๏I,?XI?M__?qo?,y?๎~?ฅ??4~?[?y?%s?ฟk?]?b?ฮBY?/4O?8์D?จ:?(]0?๛T&?}?g*?ไ 
?~~?ฝ๒>?๖โ>ฏ7ิ>์Lฦ>N1น>$?ฌ>ฟKก>ษq>G>ฤ>"ฝs>Ec>bS>สE>g7>ิ๎*>b*>2>ฯ๘	>?q >ร#๏=ู?=n4ฯ= ?ภ=Dณ=ง==ธ=7ฐ=?ฒz=Pi=K!Y=_J=<=V /=ฅ?"=s=?	=??=?G๔<GSใ<ีำ<ง?ฤ<R2ท<มzช<ฅ<ภก<b<๗ฐ<ม๐m<๙k]<ผN<ฌพ?<ำn2<&<m<.ส<ฮ<aS?#ฝIน|ฟตซผAg?ฉ~nฟ?ฬศบพ?/ค>ึ๕พๆE?><~ฟfN?ๅ`ฎผ^ฟุD?5่>าWnฟpnรพฆz^?๕O-?n?๘พเฐ~ฟ๎หภพI6???|t&?ค?ีฝฆ$Bฟฎ?ฟ)
Lฟฉซพ!ฅ>1V4?วดt?ว|?ดU?ีช?)`b>xข๔ฝnูพ๒บ+ฟ๖๛Xฟbtฟพ|ฟ,ร|ฟ?oฟ
YฟY=ฟuฟk ๚พ\1ทพcไjพ(VูฝVย%<I๒=ล^>ะท>lว>ฉ4ํ>ใf?mW?ท#?ึC/?ฺ9?๑B??J?ฏxQ?ชW?Iี\?Tsa?ye?_๙h?l?ฟชn?๗p??๖r?๔ฒt?;4v?}w?jคx?ิy?ฮyz?ฤ6{?ฺ{?|h|?{ใ|?N}?eช}?c๚}?ฐ?~?ธ{~?ทฏ~?ย?~?ล?%?ำB?+\?r?"???ฃ?4ฐ?ๅบ?(ฤ?-ฬ?ำ?#ู?Y???โ?รๆ?%๊?ํ?๏?ฯ๑?ถ๓?[๕?ษ๖?๘?๙?๚?ั๚?๛???ข????z??ะ??เ??า&ฟlg4ฟไ๖m?ธ้ฃพ*(พ?/_!ฟฝ?๘Nพ๕งจพ8จ^?*nฟจ>r์[? 4ฟpฟ`I?`ฃ2?lฟ!uฟH๙ิฝอV?๙!g?3>ฑฉ&ฟ ฯฟข2?ฟ่1พฌ??ฒi?๘|?ฑซE?ณ)ภ>ส?ฝ๏๙พ|Jฟ?wuฟวฟimฟฝFฟ?ฟJxฎพjืฝB๕=3สฅ>์M ?J7&?าD?;?[?คl?ถw?ไ}?r??ภผ~?รัz?ชาt?\Am?{d?[?ฑQ?lฬF?x<?๋52?น"(?๗S?Nู?ชฝ?0?z๕>ศปๅ>ีึ>ฤศ>ฮป>ลฏ>๖Wฃ>"]>>r>๗แv>9f>RV>(คG>ุ๐9>)->>!>ท!>ชฦ>P >E๒=ฆแ=๒๋ั=ฎdร=X?ต=Dฉ=L=8=<u==?}=al=๚๛[=uนL=>=sM1=์?$=ท=พไ=๑๙=s~๗<ฮPๆ<?Tึ<ฃsว<Mน<๋ธฌ<oป?<๛<ศ0<<.q<ฤU`<รยP<DB<ึว4<ื:(<ไ<ฎ<J<Zฟ?ผf้RฟฎึBฟ๑ก5?;ฦผ>อrฟ?0w?kHTฟ?ปF?ทฐZฟiรz?Rถqฟช?>ิฏน>ํไ|ฟh?j5?ใ?HฟHฟ^7?QนT?ตใพฎ~ฟฐฟะ?>ห|?jQB?(=ภ;*ฟ;ำ}ฟธ~\ฟฆัพ
>!ฌ"?Im?ท7?'_?%$?'Z>6วrฝฟพ!ฟIRฟจญpฟิ~ฟ๘(~ฟ5rฟศ]ฟNณBฟ๕#ฟํ!ฟศPรพษ:พ?พ;ฏ*ผ่(ห=ป๖L>ฺข>ทOภ>๑ญๆ>#~?;ย?ษJ!?๎?-??ศ7?A?ฮ*I?ZGP?๓~V?์[?จ`?nศd?ฌ_h?Zk?6n?ฒp?br?็ft?=๒u?>Iw?ฤrx?วty?|Tz?j{?พ{?0P|?mฮ|?ะ;}?}?ฒ์}?ำ3~?qq~?ัฆ~?ี~??~?ฦ?ั=?ีW?]n?แ?ว?jก?ฎ?น?ย?ฮส?๏ั?ุ?t??โ?ๆ?้?์?-๏?n๑?b๓?๕?๖?ฯ๗?่๘??๙?ฎ๚?e๛???????i??ย???j๏คพ๘?ฟHBร<ถvn?ขขZฟRๅภ>Q๗7ผ฿๔ฝmuฟ<ฟข>?ฃ5ฟ=?~?G๎8ฟตตTพี{??~ๆพ:RฟH ??[?์ํฐพMpฟ7๖พ8?๙๔v?jSซ>โฟ}ฦ{ฟQ"Tฟ๋พฯ>i$]???q๎S?ี|่>ฦ8;uMืพฅ>ฟุ;pฟ๕?ฟrฟมNฟ๖Lฟญฤพขพ%ฅ=a?>[h๑>๚?
ฒ??๕0X?ij?v??}?เ?-??ฐ{?v?6ตn?,f?dห\?๖?R?ึฅH??Q>?
4?ฝ์)?g ?ป?nX?@?กc๘>~่>ฦqู>ฺ:ห>*ีฝ>{:ฑ>ncฅ>แG>8฿> >*z>ช๛h>NY>13J>T<>!c/>Q#>->i>ฮ>'g๕=Upไ=]ฃิ=H์ล=\8ธ=vซ==ึ=::=สค=ro=ฃึ^=aO=t?@=3=1$'=๚=ฟ=ใณ=ต๚<SN้<จู<
ส<Fผ<๗ฎ<ฟัข<6<?<ฐ5<3t<?c<ษxS<]สD<ู 7<!j*<[<ื<?S	<?[ฟnZrฟข:bํ?ฌ<บพ?+ฟ?"m??๛ฟC+~?๎ฟAธu?e4ฟ	ม=1?}jzฟภ๐H>Rd? `ฟ?กNฟซj?ฃ:p?;ฝ|ีsฟHไ1ฟฟแ>
>q?Y?C9>Mฟpภwฟdjฟ.๗ ฟ\ ]=ด?d?ะ??๏Ch?ขู*?^็ฐ>
-:[คพ๊๔ฟฟJฟภzlฟ7}ฟ*ฟuฟ ยaฟม?Gฟ:ฌ)ฟz	ฟผOฯพฦ๐พ|พ0?ผ.ค=\;>f>T๛ธ>เ>\?7'?Y??็6+?n?5??x??ใษG?O?$qU? [???_?ชd?ฤg?๘j??ภm?,p?ฉFr?ไt?iฏu?Ew?{@x?-Iy?ฏ.z?ฅ๕z?ข{?7|?น|?U)}?}?ำ?}?ฯ'~?	g~?อ~?=อ~?T๖~?๋?พ8?pS?j?~?์?๑?๓ซ?6ท?๗ภ?jษ?ปะ?ื???Mแ?kๅ?๛่?์?ฝ๎?๑?๓?ส๔?K๖?๗?ธ๘?ฒ๙?๚?F๛?่๛?t??๎??W??ณ??ผbใพEsฟuฌ%ฟ็ฺ<???>ส}ฟ96\?งฯฟเP?>ดn๕พt2?+Irฟํhr?6Zฮพ๙ฟi}?ศพssฟ?:ณ>ฤeu?ฦn์ฝ?๓|ฟX ?พy๔??~?ฟv๘>ขิวพrฟทMeฟBฝพัH>ืN?บ?ต[`?๗?:นฉ=.ๅณพ%E2ฟะ?iฟไฟปvฟv^Vฟ?%ฟ?Qูพ>3AพV?)=
 >ํแ>๓?ล:?/[T?ั[g?^t?3|?&ฉ??{|?"*w?p?มg?ฅ^?๊งT?fxJ?&@?ถู5??ฒ+?ฤษ!?จ/?)๑?ฉ?J๛>(?๋>0?>ฏฏอ>_%ภ>Agณ>%nง>2>Tช>@ฮ>บ)}>๊k>คอ[>ไมL>๋ถ>>๔1>๎d%>~ >b>ฒ|>?๘=ๅZ็=ฐZื=ฮsศ=Pบ=yงญ=ฏก=ml=1?=pJ=	r=Eฑa=	R=dwC=ก็5=rH)=:=}=ำm=๋?<ึK์<ๅ?<กฬ<>mพ<;5ฑ<่ค<pu<7ฮ<Iไ<Uw<Z)f<ฮ.V<6PG<?y9<k,<ั <,w<ด<Z^eฟซGพ~)C?%ิ,?ซfฟXว>D?5Qฟ"Sg?ซ`ฟ7?ปTฅพ!คพIj?ไ้Vฟw;พร}?Uพfอoฟา>ะI~?ง><ฯ^ฟQฟต=๒ี_?[ฒk?dฃ>าฅใพ4?mฟjtฟฆ๐ฟบl<ฝT๖>-Y??ญo?9บ7?ญตฯ>ป๕{=rพ{๏ฟำๅBฟหgฟ?f{ฟวฟNwฟ{บeฟ
ัLฟT=/ฟด๘ฟ9,?พyพส/พฒอRฝคIz=ุ()>Z>ทฑ>y|ู>\5?>z?|จ?ะ()?5/4?๒แ=?"eF?QฺM?C`T?๒Z?_?ร`c?ฅ&g?oj?๐Im?ลo?ึ์q?๊หs?พku?ิv?x?y?fz?vิz?X{?ฅ|?ฃ|?}?\z}?วะ}?ค~?~\~?ซ~?Tล~?z๏~???3??N?ดf?>{??p?ศฉ?Vต?Wฟ?ศ?ฯ?ึ?ฃ??เ?ปไ?c่?๋?K๎?ช๐?ธ๒?๔?๖?`๗?๘?๙?f๚?'๛?อ๛?]??ฺ??F??ฃ??}o~ฟขPฟกE%>ลv?6๎พม:)ฟปT~?หdpฟOX?Zฟีs?๏~ฟบ|;?ทผฌNฟัb?ไญ(>บ๒ฟ}gซ=,ฺ?ใ๘=	3nฟู(ฟ๎ะ>8?~?h?Rzพidฟวfrฟaฐ๖พไ=>ร<?F}?๔ืj?|ฯ?Wป&>zพ%ฟEมbฟ~ฟำ-zฟ e]ฟ`F/ฟ๎พ
lkพ$?;จช_>โ/า>}?!b5?ฆ\P?ร}d?ํ}r?!+{?V?ฦ?ใ1}?๋;x?Zqq?hJi?ล+`?|hV?DL?๔A?ฆค7?mu-?#?ื?ี?f?ฬ.?>?ํ>าค?>#ะ>itย>ต>xฉ>>๐t>{>R&>?ุn>^>@PO>A>ึ3>x'>ซ๏>/>ภ*>๓ฉ๛=VE๊=้ฺ=@๛ส=4๎ผ=าุฏ=Qชฃ=๚R=!ฤ=๐=๖u=฿d=ฑT=P๐E=ณ48=ฑl+=w=Zu=ย'
= =VI๏<tญ?<8ฯ<6ึภ<bsณ<]?ฆ<ชf<๎<โ<rvz<%i<ำไX<ึI<฿า;<ตศ.<Gฆ"<[<iู<ตโฝ่e?วค|?-wฝ*xtฟ>@?z้ฝQฐพ<	?GฟYi>0?=วO.ฟ๏?ถฟถeํพฝ|?฿คผฟ๐%=d~?ูป>l@ฟทiฟพ+ถฝBPH?ซ;x?n4็>วฆคพrS`ฟI{ฟa)-ฟ๖?พฯห>XคL?n|?3พu??ตC?ีฃํ>J3๙= Yพ๗ ฟE:ฟขbฟ$yฟึ?ฟัyฟoiฟๆQฟง4ฟ}ปฟHไๆพ5คพFพ`ฝฯ,=ข,>Nz>R1ช>jาา>z@๗>เ?FO?น'?_[2?ะF<??D?งL?TLS?"Y?9^?ฝฉb?Rf?ไi?sัl?h\o?๊q?๚|s?<'u? v?ฺw?P๐x?ฃแy??ฒz?8h{?h|??|?ช}?๑i}?ย}?Q~?ัQ~?l~?Rฝ~?่~?๙?f.?zJ?ฬb??w??็?ง?oณ?ฒฝ?ฦ?Fฮ?๐ิ?ถฺ?ต฿?	ไ?ษ็?๋?ืํ?F๐?b๒?5๔?ส๕?(๗?X๘?^๙?B๚?๛?ฒ๛?E??ล??4????W@!ฟKณผษ6[?็๖$?5^ฟIษวผ๊8?qิwฟ??ฒ์ฟ-|?{Vฟ4nว>๑&น>ฉ?vฟH.?ศ4ํ>ฐpvฟ7qAพ7ศy?๕ญณ>ฃไSฟฆLฟฒh>ค๏v?้9>?งพฝภQฟ๗1{ฟ})ฟ"`=ำใ(?ณJx?๖Ks?๊๎*?
w>ํTพL๚ฟZฟฒภ{ฟี|ฟะcฟT&8ฟ1ฟืพฝใ;> 4ย>eS?H?/?6L?นya?]zp?ฺz?y็~?ศ๎?ฏำ}?M<y?zนr?๗ฦj?ฑฬa? X?ผN?TฝC?โj9?4/?$3%?ไ{?l?s	?$ ?ฝน๐>ง;แ>า>Eยฤ>๘ฝท>Iซ>q?>?>Z(>sท>฿ฦq>Ga>C?Q>ๅ{C>๏6>)>ณ?>๔?>ตุ>ห?=ง/ํ=	ษ?=อ=Iฟ=
ฒ=่ดฅ=9==ซ=ูคx=rfg=YW=7iH=ม:=์-=ฒ!=5P=ฏแ=\,=ีF๒<ุuแ<ฯั<,?ร<ฑต<ซฉ<ใW<ฅk<{A<?}<๎?k<ุ[<ๅ[L<แ+><?๗0<ฝฎ$<ี?<<ไิF?M๐?p7?รCฟ<?พ์?t	1ฟ_Q>wุa;2าฦผ[พ1พหย?
ศkฟยฌn?บพv;ฟU?b?\>d{ฟaQ`พ?ทo?+จ?๙๒ฟzMyฟศพQ+?ๅ~?	?vEพ|UOฟฏZฟฑb@ฟฃcyพ;H>>??hx?Jmz??ฝN?ฤH?iม9> พฝ้พxี1ฟ[]ฟpvฟ%ำฟฑ{ฟข?lฟVฟห้9ฟธaฟ๕u๒พQฏพ^ \พHJฝฝววป</$>ืi>ผข>Eฬ>V@๑>?4	?ห๐?ฑ?$?๘0?ง:?:C?_K?Y5R?p.X?e]?๐a?!ๆe?EXi?Wl?u๒n?ไ5q?-s?ๅแt?๖\v?าฅw?รx?eบy?ืz?ปJ{?ู๋{?{w|?z๐|?PY}?%ด}?ึ~?G~?~?5ต~?แ~?ไ? )?๊E?ุ^?pt?$?V?\ฅ?ฑ?ผ?#ล?อ??ำ?ฦู?ๅ??Uใ?-็?๊?bํ?แ๏?
๒?้๓?๕?๏๖?&๘?4๙?๚?็๚?๛?-??ฐ??"????_?>ํ็I?๚ฝr?ๅืฝ0>yฟjฉ?ๆ >U|'ฟ#Y?ฯว`ฟwJ?ฅฟโฐผnะ0?$พ~ฟ]กห>JF7?ดพWฟแๅพCc?๒ฃ?gO/ฟ฿กgฟJ8=Eg?0ูW?กฯy=8j:ฟOฟซ.ฟm๑5ฝH?ธำp?ฅy?hฦ:?ฝYฃ>๛?พ๊1ฟ็tQฟ}xฟรฐ~ฟoiฟ@ฟฟDพ?ฝE'>๖?ฑ>b|?lq*?`่G?5P^?.Tn?hฝx?V]~???๎`~?7+z??๒s?-7l?Vcc?4ะY?aฦO?QE?`,;?ผ๎0?ใ&?%?้ฎ?ษ
?~๗?ซs๓>จะใ>xี>๏ว>โ็น>ฐญ>ถ์ก>ฅ>ษิ>?H>5ดt>d>ํkT>๖?E>I8>?+>อ>;ส>
>๖ >ุ๐=฿=ๆ	ะ=ษฃม=];ด=tฟง=๛=ํM=@;=ดต{=?@j=Z=โJ=หฮ<=$ต/=๊#=+==กว=RD๕<:>ไ<zfิ<!จล<ญ๏ท<๘*ซ<I<[:<๐<ฃ\<ธๆn<?P^<ฝแN<ใ@<G'3<3ท&<)$<ำ^<s?b?พกขพึ~ฟsทi>:H?๏ั|ฟ๙A?tพฟD๕>*?ฟ[?น๐ฟ!9?นส=?แjฟTผ2??อ	?๒dฟคw๊พ็S?ส:?xฺพQะฟYD?พ~ค	? ?:u/?๖wxฝ7';ฟPฟฟัcQฟตฅญพ,มb>/?r?	ด}?รรX?ศ/?;v>Eโฮฝฺeัพปจ(ฟ๎VฟLsฟAฟ;}ฟ1pฟqZฟห?ฟQ๊ฟT฿?พ'ํบพล#rพ็ฝ	z;๋!ๆ==PY>?=>f]ล>75๋>๘? ?ษเ"?ฆ.?9?! B?&J?XQ?#8W?(\?W5a?Ce?fสh?*?k?@n?ฤุp?9?r?ทt? v??pw?>x?ฌy?gnz?โ,{?๛ั{?a|??|?|H}?ฅ}?4๖}?<~?x~??ฌ~?gฺ~?ป?ห#?KA?ุZ?๙p?#?ผ?ฃ?ฏ?Vบ?ฌร?ยห?ยา?ำุ???โ?ๆ?๘้?์์?z๏?ฑ๑?๓?E๕?ต๖?๔๗?๙?๗๙?ฦ๚?z๛???????u??ผๆw?ิv?ซพ>lWJฟ-ฟะ๋z?฿ฎ๑พี?ฝุ?>Xyฟๅั>คโฑฝฉ?พ?+j?dๅdฟD=Sลe?Z&ฟ_0,ฟlญ>?พf=?9ฟK?xฟฺพ๛๘O?ฟk?ม|Z>`rฟSฟ๖}Dฟฌ#พฝ๒๗>ผ๔f?ี}?7I?แษ>nศqฝ?๑พภvGฟ?QtฟษฝฟฤnฟฬHฟชฟฬฃณพำ?๋ฝฮ$โ=/ก>e?>ศฟ$?DtC?ป[?ฑl?๔Ww?ฝท}?V๘?ู~?{?su?๙m??๏d?7w[?๏|Q?w=G?้<?ฅ2?ี(?ษฝ?F??f?re?R+๖>าcๆ>otื>dZษ>ัผ>Lฏ>Wิฃ>บั>ฯ>ดุ>กw>ธพf>=๙V>พ?H>:>{ฐ->Pผ!>c>U4>>๊๓=๛6โ=า=z?ร=lถ=๕ษฉ=o=ษ=ะเ=ฦ~=m=ฉ\=๕ZM=ะ?=Zู1= %=ๅ=U=ๅb=ฬA๘<็<n?ึ<ศ<า-บ<DAญ<S:ก<	<ซ<Xํ<ะq<แa<gQ<ๅ?B<V5<ฉฟ(<}<!<r>^ฮพ/mฟำฟiกT?๖J>"ฏaฟ้{~?>fฟ_sZ?ไiฟR?กAgฟเฮ>EHๅ> cฟdมแ>ืtB?Ep=ฟ่ะ*ฟฺ:,?/g\?e๏mพ๗}ฟนFฟiศ>ฺz?HH?่ฉ=$ฟืป|ฟ๚_ฟm??พ๛>มC?kBk?น?วปa?๕w ?ะ็>ใ8ฝ5ฌธพฟฝhPฟรนoฟxK~ฟ<o~ฟฬ๊rฟ ^ฟs๑Cฟ8T%ฟ@ฟ-ฦพ~พ?mพแzผี่ม=?นH>๔ต>'พ>cๅ>bฬ?Z$?ฟ ?จฤ,?\7?Mฌ@?XืH?U?O?'?V?่ด[?๚w`?+d?ๆ:h?__k?สn?zp?ir?ณTt?rโu?;w?โfx?yjy?Kz?ฌ{?ฬท{?_J|?cษ|?r7}?ฮ}?k้}??0~??n~?ฐค~?4ำ~?~๛~?d?<?หV?wm???ิ??ญ?กธ?1ย?zส?ฆั??ื?>??ๆแ?๏ๅ?m้?t์?๏?W๑?N๓?๕?{๖?ย๗??๘?า๙?ฆ๚?]๛??๛??????e??\ฒ;?B?น>j๎พ๗A}ฟํd">l?`?ทhฟ"g๓>น|ๆฝ์.ฮผ$ฝะSท>ึBฟ?ํ??,ฟื?พ{}?ไโอพู Yฟุu?ฏ`?ฉฅพ)ไฟไฑฌพI2??x?ํฌธ>ภ5ฟJzฟนJWฟ|9พyฦ>?ฦZ?ธา?ฎ&V?๎๏>๐u<{nัพWก<ฟAoฟี๛ฟDEsฟZPฟ_๔ฟDฑวพฯพเญ=ค๛>Cำ๎>้?ฃฺ>?ฺW?9กi?ซำu?ม๖|?8ู?=?]ิ{?'9v?H๒n?~qf?]?V,S?น๔H?๎?>?ZX4?W9*?สZ ?|อ?C?๚ั?ซเ๘>๕่>ไแู>กคห>ฤ8พ>ฑ>Sปฅ>J>h,>ำh>@z>ๅyi>2Y>=กJ>ฒบ<>่ย/>ๅช#>md>?แ>ุ>ฺํ๕=อํไ=8ี=Yฦ=ฑธ=kิซ=ู์=ื=Y=ฆ๋=?๕o=lQ_=ฮำO=าhA=?3=S'=นเ=l=(?=D?๛<๘ฮ้<`ู<zส<๕kผ<Wฏ<+ฃ<ลื<CM<~<Iบt<ๅผc<jํS<ๆ6E<ู7<ศ*<ั์<<ไ<.ฟnฟ;bฟQ{>กย|?5ด๔พ๚Yีพบ8a?ฅ_~ฟ=๋?pwฟึo?์&ฟปภ<หา<?่ฎvฟM>?cj?ะฟง]Uฟฦิ๕>s?ย๎ผ?pฟ ถ7ฟ p>bฤn?เ?\?vQ>้
ฟZvฟพ๛kฟ๐ฟe==สD?฿^b?ฐ๗?i?๗-?๏"ถ>?ฦ8<!qพ๕!ฟ็uIฟูนkฟ๑|ฟLฟuฟqbฟซดHฟa*ฟฬ
ฟ|Pัพคๆพu=พFไฝ?=๐8>๔$>ใพท>!??>?ถ??ุ?*?้ฐ5?ว4??4G?R?N?CU?JูZ?ธ_?i๗c?ลฉg?&แj?ญm?<p?ฅ7r?ฺt?คu?tw?๘7x?หAy?H(z?๐z?L{?h3|?}ต|?5&}??}?z?}?ว%~?Fe~?F~?๋ห~?/๕~?ํ?โ7?ฑR?๊i?~?q??ซ?ๆถ?ฒภ?.ษ?ะ?ใึ?f??,แ?Mๅ?แ่?๛๋?ช๎??๐? ๓?พ๔?@๖?๗?ฐ๘?ซ๙?๚?A๛?ใ๛?q??๋??T??+M4พนฟีอyฟ!lฟๆ๚I?ฐ^ป>ฺyฟd??ฟว๔>$ฟC;?ถผuฟEวn?Wผพ่พฟข|?รF๔ฝkฅuฟ6ฆ>๚&w?กหผฝC	|ฟซฟ๚??Y?ผ?>ฆภพdfqฟฺวfฟ`รพต>งiL?ด?๐za?b	?A๐น=]Xฐพ^ 1ฟPQiฟดjฟwฟWฟ,๋&ฟึc?พxaEพW=>7>?bเ>;๐?_:?#๘S?g?ฝ0t?v|?/ข?๗?||?่Ew?=p?฿่g?9ซ^?ิT?
ฆJ?๊S@?/6?฿+? ๕!?Y?\?=?ฒ๛>๋>ัM?>?ํอ>ถ_ภ>ณ>จกง>Sb>ื>๘>๑x}>4l>ส\>qM>'๓>>$ี1>R%>V1>>ง>ฉื๘=ค็=Aื=ฉณศ=ฦฮบ=ึ?ญ=;ำก=i=?+=t=sะr=S๙a=กLR=ะตC=บ!6=~)=ป=Sษ=i=บ<?<U์<Q+?<๙โฬ<ชพ<?mฑ<มฅ<zฆ<ฺ๛<ม<คw<่rf<AsV<่G<"ต9<ะ,<%ั <๐ฆ< |ฟ2Yฟ:ํ_พ4P?*J?ใ=nฟ฿_>_่>aืGฟ๛ใ`?e
Zฟ;ล.?@พ?ธพฐn?ฤQฟน3พ',~?j#พ"rฟnq>๐่~?๒3>?5\ฟ๙YTฟฝ=Kว]?ู/m?woช>ฦ?พ+ฑlฟ Duฟtฟฏฤdฝpq๒>4X?W๑~? Vp?2๓8?ืณา>?(=Mฦพวำ
ฟ-BฟฃNgฟ3{ฟฆัฟาืwฟfฟfKMฟหว/ฟdฟภU?พ๛ถพ๒๛1พh[ฝr=Yc'>s>๖เฐ>ธิุ>{?>ำC?mm?จ๔(?E4?น=?พAF?VปM?/ET?N๛Y?๗๖^?ฯNc?g?aj?>m?ิบo?๋ใq?,ฤs?eu?ปฮv?x?ฃy?z?*ัz?}{?,|?[ก|?ย}?ภx}?bฯ}?n~?r[~?ร~?ฤ~?ฬ๎~?e?3?N?Rf?้z?ฟ?0?ฉ?&ต?.ฟ??ว?dฯ?็ี???nเ?ชไ?T่?๋??๎??๐?ฐ๒?y๔?๖?[๗?๘?๙?c๚?#๛?ส๛?[??ุ??D??ฦglฟ้&ฟ์}Lฟฤ>>๔ ?ขพ๛G%ฟcป}?น่qฟกTZ?ษ\ฟ๛ฑt?ด}ฟถS9?cm3ผำEPฟ5กa?/?1>ค๛ฟ-=์่?ธ>mฟฉ?)ฟ*Rฮ>?~?d_ ?(?uพ,ํcฟ\ธrฟP๘พ฿:>s<?w*}?k?ฯS?ผ)>~fพa?$ฟbฟ?
~ฟDzฟT]ฟ>/ฟฒ๎พฆlพp:ไ^>Hนั>๑ิ?a:5?0>P?ยgd?^or?๔"{?AS?ฃว?็6}?งCx?${q?ฑUi?8`?uuV?^QL?๘B?๚ฑ7?-?ล#?`ใ?ฌ?นฆ?^D?>	๎>3ธ?>`5ะ>ฆย>Nฃต>Tฉ>ำ)>R>>	2>ๅ๎n>^>YcO>^+A>/็3>'> ?>=>N7>Wม๛= [๊=5&ฺ='ห=อ?ผ=6้ฏ=นฃ=/a=[ั=`?=เชu=3กd=oลT=สF=ๆE8=ฒ|+=\=7=ฉ4
= =ฐ_๏<@ย?<้Kฯ<9่ภ<%ณ<๗ง<.u<qช<t<ุz<๋(i<๙X<้่I<kไ;<	ู.<yต"<ฅi<tฤพไฆฝ?ธ{?พดฝnฟrฟ~C?8พ๋งพซ?>์ฟx>q>M0ฟ??q?ฟ?โ๑พQ|? =ผึCฟ|`ู<฿ไ}?ฆพ>h?ฟeKjฟMยภฝG?x?0้>Dรขพซเ_ฟๅท{ฟภ-ฟ?๒พ)ส>?@L?%||?Bๆu?ีD?ี๎>Nษ?=m{WพV1 ฟnV:ฟ$zbฟ็yฟ๖?ฟษเyฟนiฟ?ดQฟvฯ4ฟๆฟM;็พ'tคพ8งFพฝNึ)=ฆ>หำy>ฝ๙ฉ>o?า>ษ๗>;ฬ?ค=?$'?ฅM2?ฟ:<??๑D?cL?9DS?๘Y?U3^?]คb?คf?oเi?้อl?UYo?>q?จzs?9%u?av?ุw? ๏x?เy?฿ฑz?]g{?ช|??|?}?ui}?"ย}?๔~?Q~?&~?ฝ~?V่~?ฬ?>.?XJ?ฏb?รw??ิ?ง?`ณ?ฅฝ?ฦ?=ฮ?่ิ?ฏฺ?ฏ฿?ไ?ฤ็?๋?ิํ?C๐?_๒?3๔?ศ๕?&๗?V๘?]๙?A๚?๛?ฑ๛?E??ล??4??งbRฟ@)ฟ๖?rฝปyV?59+?
FZฟ)อVฝIz=?๙Pyฟป้?ฑผฟฅ๏|?ภ?Xฟแfฯ>อฑ>ท๚uฟฃิ0?j็>Q>wฟ%Z6พ
Xz?่ฏ>)UฟCBKฟ์Zp>ฎiw?ื=?ๅฟสฝเgRฟ=ํzฟ ฟg?=Zธ)?fx?2?r?ฎD*?Lt>YWพผฟ๚้Zฟi?{ฟ2พ|ฟcฟอ7ฟNส ฟธฦพ(ฝ~<>ใูย>?50?aL?a?ฦp?Uz?t์~?ํ?อ}?S2y?ฌr?ไทj?ผa?X?จ๖M?ซC?ณX9?!"/?ฒ!%?k?/?้	?Uy ?๐>!แ>?{า>ชฤ>ชงท>Ulซ>ษ๐>?,>>Qง>ถจq>฿*a>๕รQ>XcC>๙5>ถu)>ษส>d๊>iว>ใช?=กํ=ญ?=hอ=ฦ0ฟ=๓ฑ=โฅ=์%=าv=ต=Ex=Ig=8>W=ภOH=j:=?z-=+q!===็ฯ=ะ=	(๒<.Yแ<ูดั<Y&ร<oต<,?จ<แC<Y<(0<w}<ํ?k<์~[<้AL<ณ><~แ0<อ$<Y,<(??X/@?ณ?ฌล?ษO>ฟmรฟาฅ?Z%,ฟvh>ี<?9ฝ๔พูำ?IjฟKp?ไุพ9ฟ้[d?ฤ>่{ฟVพ}p?!ล?:ขฟูxฟ\พภ,?พ~??ำ??ูJพฟPฟ]DฟNง?ฟ6duพQก>ฎ*??ฆx?๗Cz?่PN?wถ?N7>:ำ"พ-๊พพ12ฟ>]ฟกvฟไึฟโ{ฟผlฟ\๏Uฟkด9ฟ(ฟ?๑พิฏพ=[พ๖ปฝญย<"?>j>	ฃ>bฬ>[~๑>?O	?K	?Z%?0?Lธ:?๕C?~lK?ข@R?K8X?m]?๘a?ฆ์e?๑]i?u\l?พ๖n?9q?P0s?ณไt?e_v?๏งw?ใฤx??ปy?7z?์K{?โ์{?`x|?@๑|??Y}?ปด}?X~?rG~?p~?ต~?ฬแ~?#?W)?F?_?t?B?p?sฅ?ฑ?ผ?2ล?อ?ๆำ?ฯู?๎??\ใ?3็?๊?gํ?ๅ๏?๒?์๓?๕?๑๖?(๘?5๙?๚?่๚?๛?.??ฑ??#??ะ?=NdT>คา8?ึ[y?๕Bผ93}ฟjๅ?ำqh>K๐3ฟ a?่ogฟdR?ูcฟ ่ษ<ณฏ(?ฟ??>ญ๗0?,\ฟๅ;ืพฤf??
?ูล3ฟzeฟq๓q=ห3i?ฒPU?ฝ4=&&=ฟ/Fฟy,ฟกฝ"ฝ?ฟรq?y?ฅ9?b>$3พู	ฟ?Rฟไxฟฒ~ฟคiฟ=ฏ?ฟท 
ฟพ4ฝ_5>ศณ>>??+?cH?หช^?2n?นโx?ำm~????rR~?฿z?9ัs?gl??7c?KกY??O?%OE?Q๛:?`พ0?เณ&?h๐?เ?u
?Gฯ?>'๓>Cใ>มิ>qฮฦ>0ซน>จPญ>2ทก>}ึ>ำฅ>O>bt>[ถc>C$T>E>ฏ
8>ซc+>Q>ง>oW
>&ส >ศ๏=ู3฿=์ยฯ=ฐaม=า?ณ='ง=ฃ๊=D==ฃ_{=แ๐i=?ถY=ฒJ=3<=y/=๗K#=?๖=$k==`๐๔<๐ใ<วิ<ydล<ทฐท<a๐ช<<<?ภ<ฒ0<๏n<ย^<๊N<๛B@<๓้2< ~&<๏<ํึ?ฯnz?["1?HปgพA๖ฟี>ปS?8Qyฟ๗6?>8๔พPืฺ>n=ฟฟT?์ฟg@?๒q=
ํfฟ๛8?ฃล?Hhฟจ?พKW?฿A6?มcไพฟg6ำพ`?+ภ?ศ,?ฝB=ฟํ?ฟหฃOฟ้Zจพส๎l>ื0?lUs?ะi}?TทW??ฐ?คo>ฯq?ฝXิพeฏ)ฟFWฟ+ชsฟ}Vฟ}ฟ?ฒoฟญ๚Yฟปu>ฟtPฟ???พฌฏนพ6ปoพ\โฝ ฌม;บ๊=![>ู>dฦ>g฿๋>ัฮ?tะ?V#?ฺ.?C29?ญHB?ฌ@J?k:Q?GSW?ึฅ\?๚Ia?Ue?	ฺh?ร้k?n?ใp?#ๅr?sฃt?ศ&v?าvw?Kx?y?3rz?,0{?ีิ{?c|?1฿|?WJ}?,ง}?๗}?E=~??y~?็ญ~?0?~?i?a$?อA?I[?[q?x??[ฃ?ฦฏ?บ?ึร?ๆห?แา?ํุ?*??ณโ??ๆ?๊?๙์?๏?ป๑?ฅ๓?M๕?ผ๖?๚๗?๙??๙?ส๚?}๛???????-\?h?ฎฌ~?gใ?ฏ1ฟ?ฟ)ฟทฃr?ญปพ\พึ??ฑTฟฤล๙>Y+พ?ธพTb?ๆkฟฯ>f_?t0ฟ๚ฑ"ฟEF?๋5?วิ
ฟLฃvฟ,๒ฝอT?"h?aั>>$ฟฒฟ
ฬ@ฟ๓พ็B ?ๅh?(>}?ฝF?ไร>?Uฝ%๗พQIฟd!uฟฮฟPไmฟฎ,GฟO๘ฟฐพUฏ?ฝVs๏=Sค>ฌ?>ว%?;CD?[?เvl?>w?iื}?b๛?}ล~?<โz?้t?,\m?kฉd?,[?๋.Q?-๎F?ษ<?;W2?IC(?s?ป๗?ำฺ?$?็ฎ๕>้ํๅ>๛ื>F๑ศ>?ญป>M4ฏ>}ฃ>้>/4>>๑w>uAf>DV>าG>#:>xQ->นc!>ฯD>_็>ส>>O~๒=บแ=3า=ร=ถ=clฉ=Qฏ=ฏม=Q=๘9~=ฎl=บ/\=้L=Tฒ>=,w1=ม&%=?ฐ=`=>=ถธ๗<ๆ<ดึ<ขว<?ฦน<แฌ<Gแ?<3ถ<Q<ฅ<๑Jq<`<๊๓P<CrB<g๒4<sb(<มฑ<๖?,
ึ>?"ะฝz\ฟTK8ฟจ???1>ฃ>u.nฟ?y?ุ5Zฟi M?งx_ฟ๛c|?ถแnฟํ@๏>็฿ฦ>xึ}ฟg๛>mj9?แฆEฟs๐!ฟค4?๘W?s2พซ2~ฟฟ๐Hึ>{?วD?ฆฐG=k(ฟอ}ฟ)]ฟริพbโ>ถ]!?ฑl?฿S?๖4`?Z#?ษ>ปFaฝห#ฝพ?ำ ฟุQฟชepฟ๎~~ฟ>~ฟวkrฟฌี]ฟ{CฟR^$ฟฟ_+ฤพฟพJฒพBผฦfศ=6ณK>๊>3หฟ>'7ๆ>)I?.?&!!?2-?ฌจ7?*๏@?๐I?1P?๑kV???[?`?ืปd?ธTh?ำuk?N.n?p?"r?{at?ํu?)Ew?:ox?ตqy?ำQz?{?ผ{?tN|?ํฬ|?:}?v}?ธ๋}?๛2~?ถp~?/ฆ~?ิ~??~?]?u=?W?n?ฅ??=ก?๐ญ?๐ธ?vย?ตส?ูั?	ุ?d??โ?ๆ?้?์?%๏?h๑?\๓?๕?๖?ห๗?ไ๘?ุ๙?ฌ๚?b๛???????๗?d?้ฟ`?๛/?ค'พOกฟฦ๚?ผฑr?฿TฟSEญ>?่?<ฤNพg=Xq>f0ฟl ~?ตC=ฟฆ=พษโy?๏พ๐COฟฎ?๎X?tธพp-ฟ_พช::?๊.v?Eฆ>,?ฟ?/|ฟ)๋Rฟzพ2ำ>^^?แ?CS?ต?ๅ>๎1ปูพf?ฟFpฟร?ฟ6rฟpANฟญฟฆฝยพ5๏พ&5ช=๊>`๒>H` ?ม@?กmX?>j?7v?D)}?/ใ?ญ&?^ฃ{?๔u?"n?f?jฏ\?ฬมR?H?4>?ฉ์3?ๆฯ)?f๔?นj?>?w?4๘>๒Q่>Gู>ห>ฑฏฝ>Aฑ>]Bฅ>เ(>.ย>l>Zำy>.ฬh>๕ใX>ฬ	J>d-<>?/>?/#>?๑>:w>\ณ>|4๕=#Aไ=iwิ=Yรล=?ธ=Rซ=๘s=g==#=u@o=tจ^=6O=rึ@=Pu3='=นj=ก=๓=	๚<๏้<๏ุ<ตเษ<F?ป<ศาฎ<๙ฏข<ศd<@โ<x<๒ t<kc<๊LS<กD<?๚6<วF*<tt<่iๅพ>$๕พํุRฟ@ศvฟฉ\ฝ๐็?แmฃพฬฎฟๅp?+่ฟ์|?ฟMะw?V9ฟ๋N๗=Ya,?{ฟ๏u^>์=b?ดAฟJ?Kฟฑํ?Eเn??	คฝYำtฟณง/ฟิl> r?DBX?Q"0>|ฟํ?xฟี?iฟพ๋?พฌูv=4ุ?ธd?ย??ขภg?ํ*?๎ฎ>:ฟ0ปqมฅพษฃฟ:KฟgยlฟP}ฟFฟขๆtฟ~aฟษGฟ?P)ฟบฟฮพู3พ?พ:ๆ๒ผฦฅฆ=
9<>%>Frน>ำเ>?พ?Q?ี!??W+?6?r??OเG?-&O?IU?[?K่_?!d??อg?ง k?vศm?3p?MLr?ษt?ฉณu?๔w?ฎCx?๓Ky?1z?ป๗z?้ฃ{?$9|?uบ|?*}?}?ต฿}?(~?ฒg~?`~?ผอ~?ย๖~?J?9?ทS?อj?ห~???ฌ?Uท?ม?ษ?ฮะ?"ื???Zแ?vๅ?้?์?ฤ๎?๑?๓?ฯ๔?O๖?๗?ป๘?ต๙?๚?H๛?๊๛?v??๏??๛ู=ู๊>๒พ-aฟซoDฟo?ะ?`๚ฟZปI?งะ๓พมฆ>๘ำลพ ?๐iฟืhx?ต๗๑พS๖พdF?าXพไrnฟ?ฬ>lq?ฅ%พQm~ฟ้฿้พ๘ฟ?#
~?วธ้>๙ึพหtฟ;bฟIฑพ?`ฃ>e5Q?Y้?ภ^?Lฮ?7C=4๗บพ4ศ4ฟpLkฟKฏฟ?๕uฟศ้Tฟุ$ฟ?*ีพ@ิ8พKI=U> ๅ>;ฺ?[ข;? U?่g??นt?uc|?Fถ?๚u?:U|?๒v?;ีo?ชrg?3+^?qNT?๋J?%ส??ฃ~5?ฐY+?๔r!?ุ??ด??ษษ?>ธ๚>Zด๊>แ?>ร3อ>งฐฟ>๙ฒ>ง>cั>ฯO>y>G|>Vk>WC[>ศ@L>q>>>,1>#?$>ฮ>?>?'>๊๗=ฅวๆ=ัึ=๔ว=cบ=พ8ญ=8ก=s=ูฅ=E๗=5่q=(!a=mQ=๚B=ps5=N?(=$=ำ<=จ=[I?<ืด๋<X?<าฬ<๓ฝ<๛รฐ<ซ~ค<]<๓r<Z<๓ถv<@e<้ฅU<าะF<P9<+,<(7 <?~ฟv}ฟ3wฟ@๓พ4)$?ฯH?4ษSฟ?V<ข?/aฟ
r?`lฟZัG?ํฯพwพa?่]`ฟ ฝC5z?=PบพMjฟ(Lช>??|?๎ฎโ=Gปcฟบ๎Kฟฎ๗?=Rลc?ถh?ั>eก๏พ?pฟโrฟ ฟ_ืผฤ?>v[?คl?6Rn?A5?ึฒษ>6K=|๚พ$ฟช{Dฟะมhฟห{ฟฐฟฮ"wฟS๗dฟสฺKฟE'.ฟาฟุุพัHพ|e+พก;Bฝู=pณ,>่๖>่ณ>ฆหฺ>ษ`?>?q?๛)?๑4?2>?ฮซF?-N?TT?$BZ?ป4_?c?ูEg?>j?am?ูo?ค?q?_?s?(yu?3เv?จx?ฦ%y??z?
?z?{?#|?ศง|?T}?}}?ำ}?~?^~?{~?ๆฦ~?ึ๐~?)?4??O?xg?่{??๏?6ช?ตต?ชฟ?Iศ?มฯ?8ึ?ั??ซเ??ไ?่?ง๋?a๎?พ๐?ษ๒?๔?๖?k๗?๘?๙?n๚?-๛?า๛?b?????Gฟvศ5ฟฮhgฟหsฟg??ฝ`ศz?_XฝLฝPฟีฒ~?Zฟใ๒:?N@ฟธหc?>๊ฟื๎Q?ุvพง๎:ฟทn?ฬx=<;~ฟ	ห1>/5~?6/=์itฟ`ฟฬ๎>4โ?๚0?Pพคiฟ|nฟจ
ไพA=d>๗B?ุ^~?Ig?เ?ด>คพg)ฟ*Beฟ?ซ~ฟฺyฟ7"[ฟ?=,ฟKD็พ5z]พเฒy<ฝk>๑mื>t6?ล"7??ณQ?,ue?
!s?{?ชt?^ณ?ฤ๗|?%ใw?jq?กษh?f_?ัิU?ฌK?๗[A?!7??เ,?.๏"?K?V?ฮ?เ9?>ํ>ึศ?>eSฯ>ฝฐม>?ด>Eหจ>py>?>Xํ>นB>uเm>gข]>wN>IO@>โ3>%ศ&>คK>ญ>J>?๚=N้=+ู=ล$ส={&ผ=?ฏ=.?ข=หฑ=.=dd=๏t=ึc=NะS=คE=q7=ท*=o?=
ุ=Z	=ี =พK๎<sม?<ํ\ฮ<า	ภ<-ตฒ<\Mฆ<๑ม<ค<<<๓ly<h<้?W< I<ฤ;<l.<?๙!<ช^ ฟ@4ฟZ๕ฺพ`7>~?5ฎM>คฟน3?เ>ฮ=
ฟใ.?N๛(ฟี้>ำผฉฟP}?่.ฟ๗๎ธพ:?๘Q๐ฝ|ฟ&๊๑=ฦร?มI>ztKฟ`ฃbฟo๗ผ@ฟP?ธt??Cั>ฎๆนพฉ4eฟZyฟKW&ฟPตๆฝm0ฺ>;๘P?:}?โs?๊ื??Vิใ>Uuะ=ถบkพYฟไc=ฟ{edฟย๐yฟc๘ฟถyฟg<hฟจPฟธเ2ฟึฟใพM?พ๓ง>พ6zฝ/F=[#>&ฟ>dงฌ>ุี>็:๙>wม??6ฦ'??๖2?zฯ<?ptE?M?จS?'rY?\^?ๆb?Oผf?j?๙l?:o?(ฐq?<s?>u?ๅฌv?)๋w?/?x?๎y?	พz?่q{?อ|?็|?๘	}?go}?Hว}?j~?^U~?~??ฟ~?ู๊~?๙?!0?๚K?d??x??ฟ?Pจ?ด?>พ?ว?ฐฮ?Lี???๚฿?Dไ??็?4๋??ํ?g๐?๒?N๔?฿๕?;๗?h๘?l๙?N๚?๛?ป๛?M??ฬ??ยrฟ๓%}ฟ๔Riฟu๕เพzฯ?3a?0[%ฟๆ!ฏพ9e?ๆฟZw?vเvฟ๖ฆ??nฟgล?ษ>>nhฟฺขI?"จ>}ฟ-ฝฉ~??{>aฟก๊<ฟf*?>๐ซ{?ู{0?ีฯ'พขไZฟฆwฟD๒	ฟ๖แ?=vK2?R์z?o?#?R>Rwพ/ฟY~^ฟv๕|ฟขฒ{ฟx็`ฟป4ฟq๙พH๊พผ<L>ซษ>v?พ2?t)N?ฉๅb?nq?#z?_?ี??๓}?,วx?"r?jj?๗a??TW?๗6M?้B?8?ฑd.?i$?cธ?k`?j?uน?>6t๏>sเ>๐qั>๒ฏร>ๅปถ>?ช>!>๒i>ฺ`>ึ?>jp>&`>๛ญP>์_B>5>(>_๘>E&>ง>VV?=eิ๋=?=eUฬ=0พ=๑ฑ=ฝมค=W=Lถ=~ั=ก7w=f=*V=ทBG=จo9=า,=G =@s==?l=ฃโ๐<[*เ<ะ< ย<_ฆด<จ<p<V<y<๓"|<็กj<่WZ<`/K<8=<ฟ๓/<ผ#<ข>Bk>ฌา>O๗e?O?๗{๓พ^nCฟฺp?๙ใพุDไผY'>wพqU=์ท>ฯะSฟu{?ร3ุพ๙บฟRฮq?I>ฟ้ัฝK'x?r\๒>๔ย,ฟี&sฟa;พชq9??|?Hฟ?
ฒพงWฟ{~ฟด7ฟ์JพC2ด>ดME?ษz?ฬkx?์บI?q=?>ฃ^>`๑:พ๓๔พ้๕5ฟฏ_ฟภwฟ~๔ฟู?zฟNkฟTฟo|7ฟbฤฟผํพ AชพุQพอหฉฝOL=ป>
q>6ฆ>ฅ=ฯ>๔>%s
?ฆ?บ๗%?Z_1?Fi;?;:D?{๔K?ทR?!?X?1ศ]?Gb?^1f?บi?pl?๊#o?ู`q?`Rs?Du?yv?/พw?/ุx?ทฬy?ธ?z?X{?ศ๗{?ั|?o๙|?a}?฿บ}?ช~?L~?n~??ธ~?หไ~?ป
?+?H?ฑ`?	v???fฆ?hฒ?ฮผ?ฯล?อ?]ิ?6ฺ?F฿?ฉใ?v็?ภ๊?ํ?๐?3๒?๔?ง๕?
๗?=๘?H๙?.๚?๖๚?ฃ๛?8??บ??V#{พ๙T๑พ?พCฐ>{?ฉผ>เyฟว>>&?88aฟหนy?{ฟZทn?็ธ9ฟพj>๚อ ?ๆ*~ฟA8??Q?ฦlฟ่0พฤr?แYเ>aFฟ]Xฟ{ฝ>?q?(I?ฝ ฝmวHฟy}ฟหG ฟี๕ส<?s ?eu?ขv??p1?ป>Q6พฌ#ฟuWฟzฟคฌ}ฟ6fฟ;ฟ8/ฟt๋พฆฺWฝI,>gผป>	?	ษ-?^J??9`?ขo?ฬy?nณ~?\๘?พ~?y?ฯ8s?๙[k?ฺpb?ฮX?!ผN?ถrD?:??ๅ/?เ%?ฦ#?๎ฝ?
น	?|?ขั๑>ณDโ>aำ>Cฎล>ธ>เQฌ>$ศ?>q๖>ิ>X>'๓r>_b>1ไR>YpD>?๓6>ม_*>?ค>วต>๑	> >กZ๎=฿?=๕ฮ=:ภ=?๊ฒ=Dฆ=i?=~>=>=M฿y=#h=jX=วfI=ภm;=l.=R"=t=ฝ="ั=y๓<Bโ<"ูา<Z6ฤ<ถ<ผ๊ฉ<<%<?ํ<๒ุ~<บ'm<็ฐ\<ฆ^M<ฌ?<ุ1<B%<}.x?.วa?ๆ๙u?ep?3G>ซ?mฟงฆ^พตw?โฅZฟi๓>รQaพฦฟ<>E๐ธพเ00?๕0yฟl<]?๔ฝ๓nQฟTR?ฯร>tฟy}ขพKf?)ฮ!?ฟ}ฟ4ูฉพ?V?๗฿?๋ั?DพvGฟโ๋ฟ๒yGฟ|พณ>E8? Hv?่่{?fแR?ฬ์
?PR>ถ	พM๒฿พ5.ฟ?Zฟ?;uฟ๏คฟลY|ฟq+nฟี฿Wฟฆ๙;ฟไฟW๗พ"ดพ๕dพฅฮฝ5%z<ฯ๛==:c>ฝ>Fjษ>หี๎>ผ ?Y??%$?oฤ/?๔?9?3?B?ฯ?J?บฤQ?ฬW?9]?ำฅa?ฅe??i?E&l?จวn?ถq?อs?เลt?งDv?ผw?ฤฐx?ชy?z?ั>{?แ{?n|?น่|?R}?Sฎ}?ห?}?ฃB~?F~~?๎ฑ~?ญ?~?n??&?D?@]?s?๑?L?vค?บฐ?Zป?ฤ?ฬ?kำ?eู???ใ?๎ๆ?J๊?3ํ?ธ๏?็๑?ส๓?m๕?ุ๖?๘?"๙?๚?ฺ๚?๛?$??จ??=๊.?
๗ู>๖g?#_j?ฑY?Hกพฌhฟ!์A?Q{ผ8BฟQVB?Mฟ๔3?Hsะพ7ิพJ.C?น{ฟ๒J?>ลฌE??GLฟ๐ฟh[?็ฝ?	$ฟ@mฟ1ฒQผa?ต]?๛ณะ=ธ3ฟH๋ฟ๒ร4ฟี่ฝ9?Imn?Y๙z?Uฤ>?ภผญ>้ฝ#ฟเNฟ๒vwฟฏฟkฟ่ยBฟธจฟzบคพYฑฝ?H>ซฃญ>ณฃ?q๐(?!ผF?r]?ฑปm?!ax??3~?๒??~?๓gz?ํCt?Al?ฮc?ำAZ? <P?๗E?hฃ;?d1?ถU'?7???;?2Y?\-๔>ไ>ตซี>ญซว>e{บ>Lฎ>ศnข>>๑F>ณ>ๆ{u>ฌฝd>$U>F>สเ8>Z+,>Q >1E>(๙
>ี`>ฤเ๐=X9เ=uถะ=zDย=?ะด=รJจ=ฎก=ชฦ=ฅซ=๒|=ยk=ีถZ=ำK=ีk==MG0=๒$=งฉ=l=G5=h๖<'?ไ<:ี<Lฦ<ภธ<kนซ<ซอ<ธต<เb<xว<ญo<ๅ	_<ํO<%A<dผ3<๖A'<จ์:?หคg??fU?f๘อ>vดฟ๗rฟตี>'?G๘ฟ#๘Y?๙ฃ&ฟ5?ศ6ฟฅาi?Lด}ฟฯฆ%?c๎G>x!sฟาช"?9J?ํ๚[ฟยiฟาgJ?D?Vภพข๚ฟง์๑พ/๚?>ำช~?Cs6?=๘อผ?E5ฟญFฟๆUฟ์lบพ้I>Qท*?ฯp?)V~?ฬB[?pส?`>^Pฐฝ๑?สพ%&ฟใ;Uฟncrฟะ	ฟ}ฟิpฟก[ฟW@ฟ?^!ฟq ฟฌ๏ฝพ>?wพฯB๒ฝ๊ยงบP{?=}ๅT>ฺ<>๗ร>๋้>Lส?/้?ยO"?$&.?8?\ฝA?ฦI?ฆฯP?๖V?xT\?a?Ke?Mคh?ปk?ujn?มฟp?ฦr??t?ตv?ฯbw?๐x?y?'ez??${?ห{?
[|?ีื|?๖C}?ฅก}?ฯ๒}?9~?v~?หช~?~ุ~? ?\"?@?ลY?p?U?	?ข?ฏ?โน?Hร?kห?wา?ุ?ฺ??nโ?dๆ?ำ้?ฬ์?_๏?๑?๓?3๕?ฆ๖?็๗??๘?ํ๙?พ๚?r๛?????hฬ{?u฿z?+~?$l?๕๚>้Yฟ๑พ~?ฐฬฟdก๋ผ๙ป>?์พ<Tณ>๖A?ผแ๕พๅ;o?u_ฟืล<ภLj?Mสฟ๓2ฟฬ8?ดธB?q๗พ2zฟ5พVeL?๚ถm?R?n>?ฟห๔~ฟ:*Gฟธ2พ๑>ภxe?9~?MK?nๅฮ>IฝฟOํพ2Fฟqฒsฟ์ัฟgoฟฆIฟb๊ฟEQถพo๖ฝmIุ=qd>(๛>ฤ๛#?HฺB?ฤZ?ฅปk??&w?บ}?๕?่~?${?๋Cu?5ษm?b#e?ฆฎ[?ถQ?
xG?ช#=?f฿2?nศ(?ฑ๔?0t? R?ฺ?_๖>ปๆ>้ฦื>/จษ>Zผ> ึฏ>๑ค>J>น>ล><x>pg>ำOW>H>jอ:>ะ๖->ไ?!>ิ>Mm>ป>ะf๓=โ=ๆๆา=`Nฤ=๓ถถ=:ช=์F=าN=ณ=.=Z|m=ค]=?ฎM=็i?="2=ฤล%=ุD==k=Hง๘<e็<RUื<฿bศ<๏yบ<ญ<>|ก<hF<มื<x"<`3r<ใba<3ฝQ<-C<ถ?5<ฉ)<ยย8พํ้K>9k๔=ำรพ๑vฟปZฟปa?_ฝฺ=)uVฟโไ?5nฟ}-c?จศoฟO่?vย`ฟ'>ถ>+๛>ใ์ฟLฮ>sอH?X7ฟม(1ฟ๗0&?7`?>ผRพA๑{ฟ,#ฟCพ>๐x?EFK? ภธ=ๅึ ฟ|ฟุธaฟ๘โพฒไ๐=ท๑?๋*j?๓ฐ?Hืb?
."?ดะ>)ฝ๔ZตพJษฟ!OฟA8oฟM#~ฟ~ฟXGsฟG_ฟDฟ	&ฟxLฟศจวพืvพ.1พ;
ผ6ฝ=F>ฉต>๑ซฝ>Pไ>่o?5า?[v ?,?$7?ปz@?ๆซH?RุO?๑V?๎[?ท^`?+d?ภ'h?ธNk?Qn?๚mp?~r?9Kt?9ฺu?h4w?ฒ`x?ey?ๆFz?ฃ
{?Mด{?WG|?รฦ|?,5}?ี}?ถ็}?/~?ดm~?ฃ~?>า~?ฉ๚~?ฌ??;?@V??l?ฑ?ม???Rญ?fธ??ม?Nส?ั?ผื?!??ฮแ?ฺๅ?[้?d์?๏?K๑?D๓?๙๔?s๖?ป๗?ื๘?ฬ๙?ก๚?Z๛?๙๛???๒[ย>ฟ>?g๓;?3ฃบ>Hืํพ.M}ฟgI!>R?`?hฟS๓>อะไฝ[qิผ]ฐฝฬท>fผBฟ๗์?ศ๖,ฟ&?พืv}?xฮพ๐๒Xฟ,?ศ`?ฺอพใฟsฌพถU2?ๆุx?mธ>Aฟ็zฟLDWฟ5$พ๚ฦ>๗หZ?cา?#"V?J๏>C<ฌzัพฅ<ฟ#Coฟไ๛ฟตCsฟฃPฟ๕๐ฟีฉวพ2ภพรษ=โ>ุ๎>ิ๋?b?>?+W?'ขi?Bิu?๗|?Hู?y=?ิ{?ภ8v?ส๑n?๐pf?๚]?ถ+S?๔H?K?>?ธW4?ธ8*?0Z ?่ฬ?ด?rั?ฉ฿๘>)๔่>๛เู>ฦฃห>๖7พ>[ฑ>บฅ>?>ศ+>>h>(z>เxi>>Y>Y?J>?น<>!ย/>,ช#>ภc>^แ>C>ร์๕=ษ์ไ=Fี=9Xฦ=฿ธ=จำซ=#์=๓ึ=ผ=๋=ํ๔o=nP_=แาO=๖gA=ฟ?3='=เ=ว=?=&>๛<๎อ้<hู<!yส<kผ<ศVฏ<ะ*ฃ<ื<ขL<w}<2นt<แปc<y์S<6E<7<\ว*<xึlฟนผ*ฟสฯ-ฟซbnฟศณbฟJ>๚อ|?ำ>๔พ๗สีพ4Ta?ฎe~ฟ๖้?nzฟ[o?ฃ,&ฟัlล</ป<?ทvฟuล>Xj?ทๆฟ+PUฟ?๕>s?ล6๑ผณqฟ%ช7ฟ"^p>ณษn?๓๘\?DQ>็
ฟ]vฟม๗kฟ์็ฟำห=นK?ybb?ฮ๗?ลi?b-?ถ>n7<ฆzพฟ%ฟxIฟlปkฟช๑|ฟมKฟญuฟpbฟ์ฒHฟm*ฟพ
ฟWLัพโพฐ5พวฝกญ=,8>ส'>rมท>m฿>ข?xท?e?฿*?ฑ5?T5??ฏG?ฟ?N??CU?ูZ?ฬธ_?จ๗c?๛ฉg?Vแj?=ญm?`p?ฤ7r?๕t?0คu?w?
8x?ฺAy?V(z?%๐z?V{?q3|?ต|?;&}?ใ}??}?ห%~?Ie~?I~?ํห~?1๕~?๏?ใ7?ณR?๋i?~?r??ซ?็ถ?ฒภ?.ษ?ะ?ไึ?f??,แ?Nๅ?โ่?๛๋?ช๎??๐? ๓?พ๔?@๖?๗?ฐ๘?ซ๙?๚?A๛?ใ๛?q??(ษฟ๖ฯทฝ๑)ฝธณึพjqฟฅD*ฟไ๛8??็>ษ}ฟ8ูY?cRฟ๘$ิ>๕๒๎พ๛/?๛<qฟศXs? dำพ`๏ฟpฝ}?"พนอrฟ ีถ>แt?cญ๙ฝๅ/}ฟ่๙พ??เ~?6m๖>าษพwๆrฟโdฟ'ฃปพี>i{N?งย?ฺ
`?ข?ต4ฅ=ิแดพ62ฟ#,jฟ~ฟ?vฟ๋*VฟEน%ฟ=พุพ	@พF.=->D[โ>yม?ร:?ฒvT?pog?Jkt?็9|?ซ?l?Mv|?_"w?๓p??ถg?ลt^?zT?ญkJ?C@?อ5?ฆ+?ฏฝ!??#?๖ๅ?๖?36๛>ึ+๋>่๙?>oอ>ภ>๚Wณ>อ_ง>$>น>tย>ช}>๙ีk>cบ[>๊ฏL>$ฆ>>N1>VV%>ไ๒>]U>โp>r๘=dF็=Gื=bศ=มบ=ญ=Tก=_=ภ๒=?>=zmr=3a=โ๖Q=fC=tื5=c9)=6{=s=ฐa=ี?<ฯ6์<~ั?<aฬ<L\พ<v%ฑ<bูค<ศg<ม<uุ<?w<฿f<พV<y>G<Zi9<,<เผQฟ๗~ฟวฟghฟ:ฎฉพต*??G๗0? UdฟY>^t?ุpSฟข?h?(ibฟi๋9?`Yซพิ๙พv)i?DLXฟ์รพึภ|?CขพoฟA>(~??a>X_ฟWุPฟฟ=ๆe`?ขEk?;ฏก>nSๅพk0nฟ๎+tฟฮTฟ/1ฝธท๗>์Y?m*?๓}o?ซb7?๋฿ฮ>,u=FพQ>ฟCฟ ๎gฟCu{ฟงฤฟwฟweฟ฿ฎLฟณ/ฟฮฯฟMฺูพ๙@พx-/พPฝ๔n|=ง)>>ดฯฑ>,ซู>_?>?์ธ?T7)??;4?-ํ=??nF?๐โM?ฬgT?Z?P_?รec??*g?โrj?:Mm?๔วo?R๏q?ฮs?mu?/ึv?๘x?>y?u	z?aีz?#{?V|?ค|?#}?ฯz}?*ั}?๚~?ษ\~?์~?ล~?ซ๏~?&?พ3?O?ฯf?V{???ืฉ?cต?cฟ?ศ?ฯ?
ึ?ฉ??เ?ภไ?g่?๋?N๎?ญ๐?ป๒?๔?๖?b๗?๘?๙?g๚?'๛?ฮ๛?^??ฬฟ3ZฟุถIฟ1 rฟujฟุ+๔ผW~?f็	พyฬDฟ*ใ?}bฟ_E?ฎIฟถดi??ํฟค๏J?$฿ฝฺศAฟHk?tร=xฟEo>๔~?ค๓=๒rฟ@๋ฟ๗ไ>7ฏ?j๔?^พป๗gฟ่?oฟ%B๊พW>ท@?๙	~?0ฑh??ฺ?ฉ>๐พP(ฟqdฟ?}~ฟฌ{yฟทใ[ฟC@-ฟฉ้พ!bพk33<ฟg>xฒี>}?ม6?พBQ?ฝ#e?y๋r?Vh{?แj??น?A}?พ x?ฅ&q?g๔h??อ_?ฬV?ศ?K?A?W?7?ํ-?*#?py?โ-?dE?๛?>bํ>ญ?>)ฯ>๑ม>?ต>}ฉ>ฏ>X>h>ภ>ผ2n>A๏]>BฟN>=@>WX3>c'>๐>Iษ>sห>^๘๚=๋้=ืwู=ฤkส=hผ=k\ฏ=~6ฃ=&็=ภ_=?=ๆt=๔้c=เT=dE=&ฒ7=0๓*=b==ัล	=๏5 =ฏ๎<?<กฅฮ<zMภ<#๔ฒ<๓ฆ<w๘<b6<t3<ีฤy<?mh<KX<๋FI<ฌM;<มL.<8#=tโฟhกฟ"?ฆพลอ>โโ?ขฒ่=?ช}ฟ็น#?ไ?๒<fง๎พ	#?ฌฟb๙ะ>๕ร<ๆฟ_x~?fH'ฟสพึ~?,ฎฝ*U}ฟ?ธ=d?Gค>ก็Gฟ3eฟEPKฝN?๔u?ุ>ฤ็ฒพขcฟ3Czฟ[(ฟU?ฝsี>VO?ฏI}?Ot?"A?ธ็>\๔?=veพฟผw<ฟฉัcฟฎyฟ?ฟ[yฟฟขhฟฬPฟลw3ฟwฟคNไพ กพAพ>ฝSs==$*>j๒}>๔ึซ>|Mิ>h๘>๐v?๙ิ?ฺ'?sร2?Jข<?ฯLE?่ไL?พS?ซWY?Dh^?|าb?สชf?\j?G์l?ทso?)ฆq?s?~6u?\ฆv?}ๅw?E๚x?E๊y?Wบz?ดn{?|?|?ใ}?m}?ธล}?~?2T~?{~?ฟ~?๊~?P?/?{K?ซc?x?ย?w?จ??ณ?พ?ๆฦ?ฮ?-ี?ฺ๋?ใ฿?1ไ?๋็?%๋?๑ํ?\๐?u๒?F๔?ุ๕?5๗?b๘?h๙?J๚?๛?ธ๛?K??.กฟทmฟรzฟiาcฟหพ'?d)\?,ฟWfพ$pa???ฟิx?ซxฟKๆ?O?lฟบP?ฆNR>J๙iฟู๓F?(ฌฏ>r}ฟ๙ใฝใO~?ฟ>*`ฟช>ฟ>บ?{?|?1?ํ พ?๑Yฟฯxฟ๘DฟA๓=yP1?3ซz?ณp?^๖#?HV>ฑ?sพl้ฟว^ฟื|ฟถำ{ฟm8aฟ๗4ฟ^๚พ0๓พpฉผvLJ>ฆเศ>๖ ?8@2?ด๔M?Nฟb?๒Tq?kz?ะ?วเ?็}?าำx?ี2r?;*j? a?คjW?^MM? C?ฎ8?ฮz.?~$?:อ?ut?บ}?๛??>ไ๏>G(เ>๐ั>)อร>`ืถ>ฎจช>>9>ฃ>v>ต>'p>ู#`>aฮP>(~B>:#5>Rฎ(>ไ>!=>๓%>~?=]๙๋=จ?=vuฬ=dNพ=ฟ ฑ=ก?ค=7o=ผฬ=`ๆ=^w=ฐ6f=ู>V=bG=ึ9=๛ฌ,=ฑ =ฦ=๐)=\=๑<ฅMเ<฿ปะ<ฆ>ย<ะยด<6จ<&<Bซ<r<ฆJ|<ูฦj<IzZ<^OK<?1=<t0<๎ภ\?*พ>N>ผ้>ช็j?๙H?็กฟบ'=ฟาps?๒พ.l;ฉญp>f?uพฦnๅ<:ย>ฤลVฟ๓z?ลฯพ$!ฟฅup?ฦึ>?Aฟ2ห๊ฝฦew?ิF๗>ติ*ฟๆsฟฌ]Dพท?7?ฺ|?5L?qื|พdอVฟ0~ฟfฆ8ฟ=ไOพ9?ฑ>D?Wz?(ฆx?FJ?ซ?>?e >8#8พ>h๓พ5ฟng_ฟ?wฟ๛๑ฟ`๔zฟ_ykฟ@Tฟฟ7ฟ๑ฟUซํพ-าชพ
๑RพF฿ซฝbุ?<ฅ>1ฒp>nืฅ>่ฮ>cภ๓>@Q
?ํ?)?%?๐G1?ฐT;?,(D?ชไK?ถฉR?X?จฝ]?ี=b?`)f?ฦi?dl?ฉo?I\q?jNs?ิ?t?vv?ปw?๐ีx?ลสy?	z?	W{?๖{?ธ|?|๘|?@`}?(บ}?~?K~?๗~?ธ~?rไ~?n
?S+?ัG?`??u?`?h?Jฆ?Oฒ?นผ?ฝล?อ?Oิ?*ฺ?<฿??ใ?n็?น๊?ํ?๐?/๒?	๔?ฃ๕?๗?;๘?E๙?,๚?๔๚?ก๛?7??ํo็>?พๆs๖พพชEซ>wคz?ถถภ>๒lyฟ*{>}ฆ?V?aฟz???{ฟ0o?ท:ฟ/o>?ฤ?>(~ฟ฿๘?h?zแlฟพลพํ๛r?๔1฿>+ยFฟ53Xฟษ>ญฉq?qแH?ปฝ
Iฟl}ฟก ฟ๘ิ<ฎ ?๏ซu?าv?E1??O>า7พKฟไWฟzฟbง}ฟ&fฟ~;ฟ^ฟgณพญ'Vฝ?ฌ,>D่ป>ฌ	?ุ-??J?dB`??งo?8y??ด~?(๘?9~?y?z5s?Xk?lb?๙ษX?gทN?๎mD?ศ:?+แ/??%?X?ซน?๕ด	??>ส๑>ด=โ>ฤำ>จล>%ธ>^Lฌ>๙ย?>๑>ฯ>ำS>9๋r>)Xb>F?R>ไiD>๗ํ6>#Z*>ภ>ๅฐ>d	>ส >บR๎=&ุ?=ฮ=%4ภ=ๅฒ=ผฆ=C๗=ณ9=:=?ึy=hh=ฯbX=`I=g;=รf.=ทL"=n	==ศฬ=lq๓<ทโ<าา<า/ฤ<|ถ<ๅฉ<ี<! <p้<vะ~<ีm<ฉ\<ะWM<O?<&า1<nZd?Few?d`?'6u?{@q?๕eP>w,mฟ๑fพ.x?@บYฟ๐>^[พั+7>yถพJL/?Mํxฟ็ฤ]?ดu?ฝ็PฟjR?A&ย>Uอtฟ	.กพ1ef?kW!?ํ	ฟM๑|ฟ_๐จพฑ???่?๔พQำGฟู้ฟ0KGฟ๒๖พอ>ไฐ8?\Wv?฿{?๐ลR?๓ฦ
?/pQ>1Q
พ3เพ?M.ฟ๑ฐZฟBDuฟ]ฆฟU|ฟห"nฟ	ิWฟส๋;ฟ?ฟ_๎๖พBดพ๏นdพAอฝu}<๙0?=gc>^ั>|ษ>(ๆ๎>(?ั?H+$?zษ/?c:?C?8โJ?ตวQ?ฐฮW?]?ฮงa?ภฆe? !i?'l?สศn?ฒq?จs?ฦt?KEv?Kw?@ฑx?๕ชy?uz?!?{?ฬแ{?ฤn|?ํ่|?ฦR}?{ฎ}?๎?}?มB~?`~~?ฒ~?ภ?~??'?D?J]?s?๙?S?|ค?ภฐ?_ป?ฤ?ฬ?nำ?gู???ใ?๏ๆ?K๊?4ํ?น๏?่๑?ห๓?n๕?ู๖?๘?#๙?๚?ฺ๚?๛?$??๘ฌ~?#?+ถ>ั๓๛>ิc?Xa?ฟXพenฟq9?VZ๚<^rฟษฮH?Sฟ?:?/S฿พีฝ>?งG|ฟแํซ>?A?WrOฟ๐N?พr]?|S?'ฟ7ิkฟกิ::_c?D2\?๖บ=j5ฟีุฟ?*3ฟร:ฝ[ื?|o?๎ขz?ำท=??ช>ฃ฿๓ฝ8ฟ;Oฟพwฟผ๕~ฟฌjฟ)0Bฟ?๙ฟ@Hฃพ$ฐซฝ:็>ะหฎ>K!?อV)?G?Fญ]?_ไm?ัyx??~?  ?0z~?๗Wz?.t??}l?ษฑc?ภ#Z??P?๛ืE?฿;?E1?U7'?วo???๋
?K??!?๓>๏Qไ>กี>ว>HTบ>๏ญ>JLข>;b>ฌ(>น>๑Fu>2d>๑๋T>qUF>ธ8>ี,>. >$>ฤฺ
>D>ฌ๐=4เ=ฒะ=?ย=Nฉด=ั%จ=H=ฅฆ=ื=tO|=ะj=มZ=^K=/B== 0=?็#==-๒=3=Hฺ๕<ศษไ<Z่ิ<? ฦ<'`ธ<คซ<ช< <nD<#ซ<าxo<ำุ^<B`O<?๚@<ุ3<*ะ=dH??o?ำ?^?NX์>CC๒พO*wฟ=ผ>๙{0?dแฟ	T?ศฟอ฿?8ฦ/ฟ$]f?~ฟธ +?๏->"qฟำ'??^y^ฟฐq ฟPM?ป๔A?4วพ??ฟO;์พ๑?ญ๏~?4?ดฝฺ6ฟถlฟqTฟทพ}P>แ+?FMq?ณ-~?4Z??ึ?ฒ?>๎lธฝเฬพoะ&ฟฮฏUฟขrฟNฟฒ~}ฟpฟ#D[ฟ??ฟE? ฟ` ฟ#ฝพPpvพำN๏ฝ่v8ช
฿=zV>ล>ญ	ฤ>?๊>P๛?ฒ?Av"?H.?iฑ8?ืA??I?ฟใP?W?ษc\?ha?๋"e?jฎh?ิรk?rn?eฦp?Fฬr??t?v?fw?4x?ึy?gz??&{?เฬ{?ฃ\|?7ู|?)E}?ฐข}?ถ๓}?็9~?ตv~?aซ~? ู~? ?ฝ"?b@?Z?Jp??9?ชข?,ฏ?บ?bร?ห?า?ฃุ?้??{โ?pๆ??้?ิ์?f๏??๑?๓?8๕?ช๖?๊๗? ๙?๐๙?ภ๚?t๛???/|?ีแ?Aq?.x?;ธv?๓ฝ>ึJฟ็Hฟ๛z?ฬ1๒พg?ฝ๕?>@Fฟkั>Eฐฝส๑?พ>j?ญาdฟh=Dีe?ศf&ฟG,ฟ๘>??x=?ฯ#ฟฟyฟCXพํO?'k?มZ>eฟ RฟDฟีA#พ;?๗>ฬ๏f?eึ}?ฏ=I?~๒ษ>n?qฝq๑พrGฟ.OtฟพฟฏฦnฟBHฟรฎฟีฌณพ~ ์ฝโ=ฯก> ?>4ฝ$?>rC?9 [?ค
l?NWw?mท}?M๘?ฦู~?๕{?๕u?m?O๐d?๒w[?ฑ}Q?<>G?ุ้<?Iฆ2?(?พ?๘?? ?f?,๖>๘dๆ>uื>k[ษ>ศผ>4ฏ>1ีฃ>า>>gู>Oขw>๑ฟf>a๚V>ฯ@H> :>hฑ->-ฝ!>1>5>2>7๓=28โ=;า=?ร=mถ=฿สฉ=H==แ=ใว~=ศm=ฏช\=\M=ุ?=Nฺ1=%=น=JV=c="C๘<ุ็<?ึ<(ศ<า.บ<3Bญ<1;ก<฿	<l<๎<ฮัq<a<ดhQ<๑?B<W5<I@Hฟฮ๘ผpBซ>Jฬ}>พ ปmฟ-ฟ"ัT?ฝI>	aฟb~?+ฌfฟZ?ดiฟอ?ํ,gฟsฮ>๔ๅ>ฮeฟsแ>นB?2[=ฟๆ*ฟศ&,?t\?mพก?|ฟ?WฟซFศ>z?MRH?#=3$ฟผน|ฟz `ฟ๒?พWั>๘;?ษ>k?:?ฟa?ฟ} ?๕>ญ7ฝ
กธพBฟบePฟธoฟ๓J~ฟฒo~ฟ์rฟส^ฟ๓CฟV%ฟพฟ2ฦพ_	พOwพม{ผขุม=3ฒH>ฒ>พ>ฉๅ>+ห?F#?พ ?ะร,?ว[7?ฆซ@?ฦึH?ี?O?ท>V?ด[?ฅw`?แd?ฅ:h?'_k?n?bzp?Er?Tt?Vโu?s;w?อfx?gjy?}Kz?{?ภท{?TJ|?Zษ|?k7}?ว}?e้}?๗0~?๗n~?ฌค~?1ำ~?|๛~?b?<?ษV?um???ำ??ญ??ธ?1ย?yส?ฆั??ื?=??ๆแ?๏ๅ?m้?t์?๏?W๑?N๓?๕?{๖?ย๗??๘?ั๙?ฆ๚?]๛??๛?ฑซคพ???0]?฿Y???ฬ๖พ?ฟปHu<O9o?>Yฟฝ>๋ป พ๎<, >ฝ4ฟเพ~?ฑ9ฟฏPพMีz?ื่พ?Qฟ{ฟ?ธZ?ฌ.ฒพeฟ?พชw8?ำv?urช>๒@ฟู{ฟo์SฟO>พ70ะ>hK]??ฦศS?่><ฤ:ฏืพ+ว>ฟLpฟ??ฟtrฟLซNฟ&1ฟN?รพ4*พกฆ=ส0>๑>ไ ?ภ??;X?ูj?ฤ v?}?แ?๗+?ฎ{?นv?2ฑn?(f?ฦ\?ใูR?ช?H?ญL>??4?ธ็)? ??์S?๒?q[๘>หv่>njู>ํ3ห>คฮฝ>Y4ฑ>ญ]ฅ>|B>*ฺ>?>R?y>f๓h>Y>?+J>KM<>?\/>พK#>น>U>าษ>U^๕=hไ=ทิ=(ๅล=บ1ธ=ๅoซ=B=|=>5=&?=qio=ฮ^=
ZO=~๗@=3=('=]=eบ=ฏ=๛ซ๚<็E้<ัู<Rส<|?ป<ย๐ฎ<?หข<ฝ~<i๚<๓0<ษ*t<[7c<&qS<BรD<<7<๑erฟ๕ุQฟลI ฟ๛ฟ?"ZฟS/sฟฬผจ๘?บKถพุฟุัm?P?ฟ๙}?2ไฟ?v?ยL5ฟ!~ส=ฎ80?ซ?zฟoฑL>Q0d?ฟ-Nฟษ?X?o?Aฝmtฟ1ฟู>?eq?oZY?ญ7>ชฟ่ึwฟแiฟ9ด ฟPa=;ั?n2d?ํ??--h?&ด*?zฐ> ๐9+UคพbฟิJฟBlฟ;}ฟL(ฟ้
uฟqถaฟอGฟV)ฟฬ		ฟ!.ฯพๅฯพฃ=พญ๛ผjค=ษI;>U>	น>ฎ-เ>ง?.?เ?ฉ<+?6?R}??หอG?๘O? tU?ป[??_?คd?าลg?๙j?Iยm?ฉ-p?ฅGr?พt?&ฐu?้w?
Ax?จIy?/z?๖z?lข{?ู7|?Vน|?)}?ม}?๚?}?๑'~?&g~?็~?Sอ~?g๖~?๛?ฬ8?|S?j?~?๔?๘?๙ซ?<ท??ภ?nษ?พะ?ื???Pแ?mๅ??่?์?พ๎?๑?๓?ห๔?K๖?๗?น๘?ณ๙?๚?F๛?่๛? uxฟ๖dกพ@B7>ยU>wJพYฟญ5Mฟฉ?ิษ?฿ฟ2C?fCโพฆD>_๐ตพฝผ?SีfฟQ?y?ท2?พp์พ๕ก?Sมlพ๘งlฟ	ี>ทp?๎4พอ~ฟธแใพ:?ช}?ๅไ><ฺพfxuฟพ3aฟฟญพษ๓ฆ>T7R?๓?๔K]??s}}= ?ฝพ5ฟ?ถkฟ[ปฟ3ดuฟ฿oTฟ8~#ฟแัำพธ6พนษS=Sท>฿?ๅ>C?๖;?_U?+h?Nึt?ฺr|?Kบ?ฝp?ขH|?ห฿v?ฉพo?Yg?s^?h1T?>?I?Xฌ??a5?พ<+?าV!?ฐภ?ข?โฐ?ื๚>d๊>Z^?>กอ>ฺฟ>๘ีฒ>ผๅฆ>ฒ>2>^>๙W|>&k>[>๚L>o>>01>5ฺ$>->้>d>^ท๗=๙ๆ=$ฅึ=ฝสว=โ๕น=ไญ=6ก=`ํ=์=X?=ถq=๒`= XQ=!าB=ัM5=Kน(=?==n๚=ำ?<๔๋<+?<{๔ห<&ฬฝ<Pฐ<\ค<๓<fU<ฺs<ฤv<fe<yU<งF<๎?8<๘ฝvพะ๒rฟ๗?{ฟfzฟง๑zฟษ๑ฟa?4vP?จLฟ^? ฝึ%?Yฅeฟ
แt?ฒJoฟmฒL?`?พ&ฉ\พf^?&.cฟZ[ฝIy? >รพษhฟใ!ฒ>ๆ๘{?E๚ล=ึ>eฟdJฟไ
>6?d?d}g?๊T>y๓พฯpฟ๙rฟฟnฃผHณ ?ฯ/\?a?็?m?้r4?สพว>l;=พพ๕ูฟ?Dฟiฟ:๋{ฟVจฟร๚vฟทdฟ?Kฟ์ฬ-ฟsฟฯุพพ9๙)พ?<ฝW=ุุ->|>ฅณ>8?>ขม?>ฎ6?D?ชฒ)?จ4?L>?งยF?-,N?ฯงT?fQZ?	B_?4c?๐Og?j?)im?:เo?fr?^เs?~}u?๖ใv?๋x?(y?qz?*?z?ใ{?/%|?*ฉ|?}?~}?vิ}?ี~?B_~?~?hว~?F๑~??๓4?'P?ทg?|?ส??Yช?ิต?ฤฟ?`ศ?ีฯ?Iึ?เ??ธเ?้ไ?่?ฏ๋?i๎?ฤ๐?ฯ๒?๔?๖?o๗?๘?๙?p๚?/๛?ิ๛?&:ฟบ
sฟ.+"ฟjยฟณPฟชf}ฟ:ศพฒธp?:้ง=แงaฟื๔y?๗@JฟทS(?ฯ/ฟ'ณX?~ฟGo\?{bพn/ฟโ๋s?ซYz;I|ฟ
?c>๖|?ฒาW9c>wฟ\มฟ??>ไ??8้?ัฃพนBlฟ;lฟภูพ๎x>๚ลE?"ู~?^ปe?x?๖๙=H5ขพๆ+ฟๅfฟM๐~ฟ๑xฟถเYฟ{*ฟุใพโUพ|ฎถ<?Gr>๑Cฺ>ึd?8?nlR?ฬ๙e?xs??ต{? ?จ?Aื|? ฒw?๑รp?h?ฐR_?9U?๑YK?ำA?บ6?ข,?^?"?ํ??2ธ?โิ?นด?>ย์>FQ?>โฮ>iFม>wด>^mจ>^!>>?>Cฒ~>rYm>K$]>ฦN>kแ?>eณ2>h&>๒>ฃC>๊N>R๚=ยว่=ฎุ=Gฐษ=บป=?นฎ=%ข=@Z=?==ณt=ac=๓US=ยฌD=7=lT*=?==ึE	=ฉ}?<ยํ<DA?<ฃๅอ<ฯฟ<?Mฒ<8ํฅ<zh<cฐ<มถ<ฟ?x<ใg<	W<ไH<?:<ฝ/?ซิ?พFฟึาSฟ ฟ๘>?u?S;ฎ>]#ฟ๕ษ๑>q0]>๐ฟีเ@?>P:ฟUL??W?ฝื+ฟ๔จy?อ:ฟ฿iพ??สฯ-พูyฟ?o'>  ?ร>PQฟpi^ฟ$๑:๛U?๖r?&ล>KลพปฑgฟRxฟ,"ฟตยฝ้้แ>?S?
~?ะr?4ณ=??t?>5Kบ=สฬuพ:hฟaไ>ฟ็TeฟxZzฟด๏ฟ4ปxฟฐgฟs*Oฟื็1ฟอฟI๋เพ:พๅจ:พฦู{ฝT=?_ >UY> ?ญ>๛;ึ>qM๚>;?H?ุ%(?K3?=?_ตE?s@M?ลูS?Y?2ฅ^?c? ูf?+j?9m?o?ภq?sฅs?]Ju?ทv?r๔w?=y?๕y?ฤz?'w{?Y|?ื|?b}?\r}?ุษ}?ฃ~?JW~?*~?mม~?์~??1?สL?อd?y??4?ถจ?hด?พ?Pว?้ฮ?}ี?/??เ?dไ?่?L๋?๎?y๐?๒?\๔?๋๕?E๗?p๘?t๙?U๚?๛?ภ๛?m7=>cญQฟk๔ฟา]|ฟK็}ฟษ*ฟธ>w?ต์พฆ
ฟุฦv?jzฟh?xiฟญ{?ryฟ๛)?%ฮ=5
[ฟถtX?\t>ฝญฟ?๙<c่?TX5>d๎hฟฅ2ฟ?fป>ส}?11'?งชUพGS`ฟๆtฟ๏ฟ">8?P|?m?ฮ?ๆ!:>Iจพ<พ!ฟ?โ`ฟ@}ฟโzฟซ๛^ฟn1ฟ?๒พฤpuพไ{้ปึ๒V>ปdฮ>|p?(4?ฅbO?์ศc?r?~็z?1>??ั?_Z}?ฒzx?มq?[ฆi?6`?NาV?ผฐL?bB?8?_เ-?*่#?ป;?่?๏๗?฿?>โค๎>0C฿>ธะ>Nร>ถ>๔ฉ>J>Sโ>ๆแ>>o>ส1_>a์O>?ซA>z^4>ื๖'>ึe>ฑ>a>/i?=z๗๊=ีทฺ=ลห=~ฝ=ฬ^ฐ='ค=ว=<0=ฒT=MOv=?:e=โSU=`F=Lม8=๏+=? =ฑๆ==
=?๓ = ๐<}W฿<หึฯ<wiม<l?ณ<ไ}ง<W?<_<ฉ๙<บ5{<&ลi<zY<4pJ<Rb<<q{?G฿?ผ๊+พ~ศ>gด>?}ษn?,พ0
cฟ/|Y?ั1พธ2Rพส*ึ>bาพ?ฒG>S๔k>ฐD@ฟดY?=ฟiฏฟฉx?ว)M=฿๔ฟiโ?ผ ๔{??gิ>,๋7ฟB<nฟไGพเ?A?t]z??ฐ๖>?#พ&\ฟมล|ฟฒ๐1ฟส -พผSม>]kI?>ผ{?ะ?v?nF?๋ฃ๔>?>ตฌKพ๛พ8ฟ~Uaฟ?xฟV?ฟๅKzฟNGjฟtฌRฟ์5ฟOฟผช้พA?ฆพ{KKพcฝ๗s=ึ฿> bv>ชjจ>9ั>ุา๕>d=??พ?;&?๋1?ไ;?๓ฅD?ฯRL?
S?%่X?^?พ}b?af?.รi?zดl?;Co?|q??is?รu?ีv?อw?ๅx?Qุy?วชz?6a{?U?{?]|??|??e}?!ฟ}?Z~?@O~?2~?dป~?฿ๆ~??%-?eI??a?w?f?K?ง?๚ฒ?Lฝ?=ฦ?๚อ?ฏิ?}ฺ?฿??ใ?ค็?็๊?ปํ?.๐?M๒?#๔?บ๕?๗?L๘?T๙?9๚??๚?ซ๛?๚Cm?ข๏ผ")ฟ'ลMฟ2#ฟ:.ฝ)ๅY?ย&?ฎ]ฟฝs9:?TDxฟฯ??โฟx[|?4Wฟ?ณษ>ท>9ถvฟ/?ถ๋>๛ซvฟK>พพ๑y?@aฒ>0ATฟ
-Lฟ=฿j>?w?็ๆ=?
ตมฝOอQฟฃ{ฟ2?ฟ=C )?ๅ\x?@6s??พ*?ะv>\Uพ{$ฟRชZฟใศ{ฟ?ฮ|ฟปพcฟ?8ฟ7ฟ`พฟฏฝjt;>6cย>3g?H0?yBL?ยa?sp?gz?ไ่~?s๎?๓ั}?y9y?ะตr?ฏยj??วa??X?N?$ธC?นe9?๑./?0.%?w?ะ?		?๏ ?มฑ๐>4แ>฿า>ปฤ>ขทท>U{ซ>??>ั9>t#>?ฒ>Nพq>?a>หึQ>์tC>n	6>)>ู>ฏ๗>หำ>๗ม?= 'ํ=ม?=8{อ=#Bฟ=ตฒ=๏ฎฅ=๐3=?=ฺ=โx=^g=ฮQW=๛aH={:=ฉ-=?!=ศJ=ข?=ฉ'=>๒<ดmแ<๒วั<8ร<๙ชต<ฉ<4R<[f<<<ด}<i๔k<๊[<TL<%><หBภ>ไ?	ู??DL?๐๙D?Y??a?<Bฟpฟฏ???จ/ฟษy>ภน!<-ศ๗ผH!,พฆ
?YLkฟfo?ชพเี:ฟงKc?๎๓>m{ฟoh]พ๓๕o?I?ธmฟฯ,yฟึพญ+?2ฺ~?ฉฑ?k?Fพ๏Oฟ}Tฟ-@ฟAxพงฬ>หฝ>?/wx?az?หN?>?i9>๊/!พwา้พฑ๏1ฟx]ฟ)yvฟ9ิฟฌ{ฟ๛ิlฟ Vฟฅฺ9ฟ\QฟXT๒พ๕pฏพั฿[พฬฯผฝฝ<?X>เj>uาข>๓0ฬ>๔Q๑><	?ภ๗?ุ%?d0?Fฌ:?iC?AcK?8R?=1X?wg]?น๒a?๛็e?โYi?ํXl?ฌ๓n?๒6q? .s?ฑโt?ง]v?lฆw?รx?ูบy?;z?K{?%์{?ผw|?ฒ๐|?Y}?Pด}?๛~?!G~?*~?Mต~?แ~?๖?0)?๗E?ไ^?zt?,?]?cฅ?ฑ?ผ?'ล?
อ??ำ?ศู?่??Wใ?/็?๊?dํ?โ๏?๒?๊๓?๕?๐๖?'๘?4๙?๚?็๚?๛?Q?ภH?A>ฬพi=?nอ~?ทํึ=หูฟb๑>มค>?Cฟฅj?ภdoฟY\?ีฎฟgต=?๊??ฟ~๔>โ'?ณaฟกปรพแj?
;?fว9ฟcraฟม?ด=๘k?๐ฐQ??ฉ<ถ?@ฟ้ิ~ฟ_(ฟ30iผ?฿s?์0x?/ร6?ก>๚ปพอ ฟํSฟ0nyฟYH~ฟ(hฟฃl>ฟ?ฟHไพjฝา>`Aถ>ณI?u่+?AI?'_?e็n?ศy?~?u??๚=~?l๎y?Vขs?ุk?๛๙b?'`Y?RO?๎
E?Gท:?T{0?nr&?ฑ?ฺE?-;
??[ฝ๒>๖#ใ>Lbิ>uฦ>Wน>ฉญ>mก>>วd>u฿>I๐s>Lc>มS>p>E>Bด7>+>+L>Q>'
>T >ดV๏=Lส?=`ฯ='ม=จณ=ห6ง=ม?=|ื=?ฬ=q่z=๏i=ทOY=<J=พ4<=ฤ%/=z?"=?ฎ=(=\=|๔<๊ใ<นำ<ฦล<Yท<<ช<ว<Wม<v<ฎ็<ฌ#n<[]<ิ8N<ด็?<jถฟฺ?Li}?้}?X?ึfI??ลฝ~ฟื=ฎรa?Nbrฟ+ึ$??๑ฬพgต>Pฟ}sJ?g?~ฟ๛DJ?ยฬn; ๋`ฟะCA?@๑>lฟขหพMf\?!0?@๒พฎฟwฦพ5ฺ?ใ๑?Y(?4pรฝ๓ธ@ฟ]๙ฟ#+Mฟ??พ฿?z>aB3?๋It?ฦ๛|?<>V?ป?s~f>ำ์ฝผลืพ+ฟXฟ)tฟgqฟเ?|ฟH;oฟ?TYฟฑ=ฟมzฟL็๚พu๗ทพฝdlพ1?ฝZb<*๏=eค]>ฐ4>O"ว>โส์>ม7?-?ทn#?6#/?)r9?ยB?อqJ?deQ?ะxW?ฦ\?fa?ๆme?ฉ๏h??k?iฃn?:๑p?w๑r?&ฎt?0v?เ~w?Gกx?y?swz?น4{?ฦุ{?๓f|?'โ|?่L}?eฉ}?๙}?๐>~?{~?(ฏ~?E?~?Y?1%?B?ๅ[?โq?ํ?k?ณฃ?ฐ?ศบ?ฤ?ฬ?ำ?ู?J??ฯโ?ธๆ?๊?ํ?๏?ษ๑?ฐ๓?W๕?ล๖?๘?๙?๚?ฯ๚?๛?ฯ75ฝฑQw?X?๒ ?แ8?z?ำ๛;?L๕พศสSฟ]Y?ฦ|"พ??ฯพ,?:ฟH?|?พ?fพอึP?5vฟฆuv>.P?่ศAฟ๐ฟ^DS?๔)'?3:ฟฯaqฟึแaฝแฌ\?cAb?ึ>ฐ-ฟ\?ฟDฏ9ฟ8&ฮฝY?+Ml?P๖{?B?ศPถ>?8วฝ็บ ฟ0ฏLฟbvฟ๖Mฟฯ5lฟIDฟnหฟ!Aฉพรฝํ>>ช>น?5ฎ'?UภE?gท\?;m?ดx?ไ~???m~?z?t?~ๆl?*&d??Z?qP?qZF?&<?ล1?฿ด'?u้?ณr?X[?cช?ฏว๔>อๅ>ไ5ึ>๛-ศ>
๖บ>ฎ>๎ฺข>๏็>?ฅ>?>๖!v>ำXe>ซU>หG>๕^9>ก,>5ฟ >uซ>vX>กน>6๑=rำเ=๛Eั= สย=oMต=กพจ==+=!	=๛4}=ภฅk=M[=*L=s๎==?ภ0=}$=๓=ks=z='บ๖< ๅ<=ชี<mีฦ<น<็/ฌ<๎;?<S<\ย<T <๎Rp<หฃ_<$P<eชA<ิฟฟc3พ?j_G?X1?PS>ฎฦ-ฟถร`ฟGฮ?ฐ<?ฌม|ฟ?๓i?=ฟW[/?`iHฟ_s?ึfyฟว?ฉA>}yxฟรj?VH'???SฟwฟใA?๗SL?iชพEฟMรฟu๏>อ?}?L<?r;;0ฟ%ณ~ฟะำXฟ?๓ฤพฎ'5>'?Y8o?0ษ~?G]?ม?+บ>g?ฝ๕cลพ"$ฟทฬSฟqฟ๔ี~ฟซ?}ฟัyqฟz\ฟษpAฟ
"ฟfฑฟoภพู|พ?๛ฝ๊หดปnsิ=.8Q>>วย>ภ=่>j0?`?฿ึ!?ป-?ป58?kA?u~I?P?เพV?[$\? ู`?ฦ๒d?h?fk?pRn?ไชp?dดr?#yt?v?๙Vw?ฌ~x?y?o]z?,{?;ล{?V|?zำ|?0@}?a}?๛๏}?ฌ6~?่s~?๓จ~?ๆึ~?ฒ?~?*!???฿X?Do?จ?t? ข?ฎ?น?๔ย?"ห?8า?[ุ?ซ??Eโ?Aๆ?ด้?ฑ์?H๏?๑?v๓?%๕?๖??๗?๓๘?ๅ๙?ถ๚?l๛?>S]ฟ๙vฝ>xMt?ฉ??๙?,?a?2>{dฟภอพภ???-ฟ/น=H่>Hาพl>;ฌ<ฐqฟ_ฮr?Zฟ๗ซ@ผm?:xฟx,8ฟ๖3?+ำF?์์พฤ{ฟp่IพvI?Wo?	>^ฟ ~ฟัAIฟtอ>พ%f์>rBd?บ~?xpL?เา>ฑ(ฝ๕้พ๕Dฟ๛0sฟ_฿ฟ|ๆoฟุfJฟ[่ฟฆrธพa??ฝ๚lะ=ฺค>ช๙>_#?_B?ฉ2Z?h{k?<?v?=}?๓?I๓~?ฝ:{?fbu?่ํm?Le?บฺ[?^ไQ?จฆG?PR=?|3?๕(?m ?X?z?mผ?นะ๖> ็>ฃุ>.ๆษ>jผ>๛ฐ>mHค>>>นๆ>8>TSx>Xeg>ุW>?ะH>	;>เ..>*2">?>ถ>ใๅ>ฆต๓=?โ=J+ำ=ฤ=A๒ถ=qFช=Tz=ญ~=?E==ษm=K]=พ๑M='จ?=๖[2=ฐ๛%=w=ฮพ=โฤ=-๘๘<Tฐ็<bื<คศ<ถบ<ภญ<สฐก<Ow<C<ัL<0r<<ฌa<tR<mC<ฟฆ ฟฤำmฟ๑?พ_รฑ=d<
๑พ๕|ฟ๚ฉๆพ๕j?ฏ7=พ]Lฟฅื?ฆ[sฟwZi?+tฟ๑?[ฟธ?ข>๛?x๛ฟ3พ>คM?ฯ1ฟ86ฟxB!?2๑b?ฌ๐<พh๙zฟ๓ฟiฏต>ึ๎w?M?ณyี=ท6ฟF{ฟ:cฟ~ฮ็พ|แ?=h?7Gi?eศ?ทตc?#?ธ์>@=ฝ_ดฒพ)ฟฟนสNฟฅฯnฟ~ฟปซ~ฟ6sฟf_ฟ๋Eฟฦ&ฟใฟืศพ?พนdพยขผEน=ธรD>?้>๓ผ>ญชใ> &?ฉ?V< ?OQ,?๗6?*S@?:H?๛นO?pV?ฬ[?J`?vd?xh?mAk?ฤ n?๐cp?วvr?จCt?จำu?ถ.w?ม[x?ำ`y?0Cz?k{?ฑ{?์D|?ซฤ|?[3}?C}?Yๆ}?T.~?ฎl~?ฑข~?zั~??๙~??~;?าU?l?_?y?H??ญ?8ธ?ึม?+ส?bั?กื???บแ?ษๅ?L้?W์?๚๎?B๑?<๓?๒๔?m๖?ถ๗?า๘?ศ๙?๚?V๛?ภึcฟ7#ฟ๓ศ>\ค@?ฟ=?ฮ<ฟ>ณภ้พZ}ฟg>o฿a?+ใgฟไ+๐>V์ุฝaz ฝfฒpฝ
ยด>Bฟ?ๅ?ฝญ-ฟโพDX}? ฯพฟXฟ?7`?๋พQ฿ฟfซพ}ฌ2?\พx?หอท>รฟdซzฟWฟ฿พฯว>G๏Z??ฯ?ฮV?=ซ๎>]v<[ััพjร<ฟมQoฟO?ฟ8sฟQ?Oฟฌุฟ๔tวพVพ=A.>ฒ?๎>๛?ฮ่>?W?ภจi?sุu?3๙|?บู?<?า{?แ5v?I๎n??lf?ธ]?D'S?๏H?ภ>?9S4?Q4*?้U ?ศศ?ฟ?ชอ?xุ๘>Wํ่>ฺู>ฐห>;2พ>๘ฑ>ตฅ>โ>X'>d>cz>qi>v~Y>J>๖ณ<>ผ/>	ฅ#>๖^>่?>>ๅ๕=ๅไ=ี=๗Qฦ=ธ=;ฮซ=็=?า=[= ็=Vํo=^I_=NฬO=ุaA=๗3=Hz'=?=0
=H๙=26๛<ฦ้<ู<ธrส<'eผ<<Qฏ<ฆ%ฃ<Jา<)H<My<rฑt<ซดc<รๅS<ว/E<ศt้>9Zฟฏงkฟ่(ฟู+ฟใxmฟJรcฟtอ>l}? ๚๐พํุพRb?W~ฟม฿?ฟฟ๓o?'ฟมฮๆ<?<?๐๔vฟ?>j?ฟ๐Tฟ๗>[s?&Wฝl1qฟปU7ฟr>j๏n?ขว\?เO>้ำ
ฟ?rvฟX?kฟธญฟบภ=}?|b?๘?Qi?ฬํ,?7หต>?-<Yพพฐ@ฟIฟฦkฟฦ๕|ฟ่Iฟ~uฟfbฟ{ฆHฟ*ฟ"
ฟ฿.ัพปลพn?พ?๖ฝ1=G8>๕;>จำท>ล฿>ํ?๔ฝ?$??ไ*? ถ5?@9??G?ภแN?FU?็?Z?ฬบ_?f๙c?ซg?งโj?cฎm?_p?ก8r?ตt?ืคu?w?8x?GBy?ด(z?v๐z?{?ฎ3|?บต|?i&}?}?ก?}?้%~?ce~?`~?ฬ~?B๕~???๐7?พR?๕i?~?y??ซ?๋ถ?ถภ?2ษ?ะ?ๆึ?i??.แ?Oๅ?ใ่??๋?ซ๎??๐?๓?ฟ๔?A๖?๗?ฑ๘?ฌ๙?๚?A๛?OวฝI?~ฟ_hๅพ)n<=ูวญ=8ฅพ^ะfฟ?ิ<ฟpe'?ฒn??ฏฟญ๒N?๏ ฟท?ด>้?าพฃ%?$Tlฟ{?v?,ฑ่พu?พโไ~?
HพZูoฟ\%ฦ>r?&พ~ฟ?ป๎พ7ถ?YR~?คํ>ไPาพ:tฟรcฟ3ฉดพถt?>aP?{฿?1ฌ^?pห?Aื=นพ04ฟฝ๔jฟผคฟุ*vฟสLUฟU$ฟ4Dึพฆ
;พ?+A=>๓0ไ>1?๒];?;์T?,รg?nกt?อV|?้ฒ?-z?p_|?๔ w?็o?g?ฯA^?fT?5J?qโ??ถ5?Jq+?ๅ!??๑?๖ต???็?๚>ู๊>ซ?>Tอ>}ฯฟ>ณ>S"ง>๊๊>ธg>้>#ต|>ข}k>฿g[>ใbL>D^>>CJ1>ั%>ธ>>E>>N๘=๎ๆ=ล๕ึ=ำศ=อ;บ=?Uญ=ำSก=อ%=sฝ===r=9Ga=?ฆQ=C=!5=฿๘(=*?=U=ฎ-=7t?<บ?๋<จ}?<]Aฬ<ฒพ<ๅแฐ<ค<E-<<ษฅ<ดเv<ฝe<สU<x๒F<หษ~?RrตฝZ?dฟบฟฃฟfsฟIu?พฺ,?ลฏA?๐=YฟHJ=ฑ?-)]ฟบ~o?UiฟฑC?าฤพ^ฃพยd?๛]ฟ?(พฝ๓{?ื๚ฒพPlฟ*฿ฃ>ๆ}?๚=งwbฟvMฟ{ ๊=8มb?กgi?9x>"{์พขoฟsฟFตฟูฝ?>A?Z?Y?@ฏn?้5?(Jห>ปX=[พDฟ้Dฟ:hฟuฑ{ฟทฟ7Cwฟ"+eฟLฟ?p.ฟW ฟถuูพใพ,พ@Fฝ)ะ=ฤ+>แ>_ฎฒ>(sฺ>ผ?>y่?N??|u)?บr4?E>?'F?ูN?T?ฏ5Z?฿)_?'{c?=g?j?N[m?0ิo?๒๙q?Jืs?uu?!?v??x?w#y??z?Nูz?{?I"|?งฆ|?Y}?บ|}?ิา}?j~?^~? ~?|ฦ~?z๐~?ู?Z4?ฃO?Dg?ป{?t?ฮ?ช?ต?ฟ?6ศ?ฐฯ?*ึ?ล???เ?ีไ?y่??๋?[๎?น๐?ล๒?๔?๖?h๗?๘?๙?l๚?+๛?z๔H?*ฟ๎ฅxฟนT0ฟg๔ฟห-Yฟqสzฟูตdพๆโt?อๅ=ู
\ฟ	|?`)Pฟั%/?Nฒ5ฟโี\?๓6ฟ๏ุX?N_Iพ
[3ฟ:-r?ยฯฤ<ล}ฟฝQ>แ3}??ก{<yHvฟ[|ฟ๘>??๕'?	พYkฟmฟํj?พq>งD?ฐ~?Hff?(ล?c6>แพข+ฟ:fฟหุ~ฟผxฟuSZฟV++ฟ?ไพ:Xพโ$ข<}๕o>ZBู>๙??พ7?+R?ึสe?CYs?ฅ{?จ~?.ฌ?แโ|?รw?ิูp?4h?๘m_?เ?U?MwK?_&A?๐ื6?iฌ,?_ผ"?๚?-า?ธํ?ไ?>คร์>พ{?>
ฯ>-lม>ด>ธจ>ฆ@>?ง>ป>ๅ~>em>Q]>+N>p@>หื2>&>0>!a>ej>C๚=u๗่=๐ฺุ=ฆูษ=เป=บ?ฎ=ภข=Wy=๙=x3=?4t=Ec=gS=3ีD=4-7=tw*=;ฃ=๑?=b	=9ฒ?<๋๒ํ<สn?<ฮ<<ยฟ<rฒ<]ฆ<?<๔อ<Eา<๕y<ลg<bฎW<)ตH<็?(๒>?Wคsพ?9ฟเuIฟฟ็M>sy?p8>รูฟTี?ฌ3>7ฟึด:?๒V4ฟy}?L?ฝ1ฟK {??ช6ฟ.๘ฅพํ?WฮพzฟE๗>E๘? น>^Oฟs๑_ฟ๛ผศS?IVs?ฃwษ>Bมพฺาfฟๆลxฟ?ๆ#ฟำฯฝu-฿>็qR??๋}?ๅ2s?๔v>??^เ>?*ย=๑9rพญฟC\>ฟR eฟY5zฟ)๓ฟ6฿xฟ7ฯgฟ$xOฟ_@2ฟC+ฟ๒ชแพใ๖พO<พฝYO=ย9>6ำ>อญ>๒ฮี>?๋๙>???\?ๆ(?5-3?<?<?SE?F,M?"ศS?#Y?ฤ^?฿๛b?ัฮf?ด"j?m?do?นบq?h?s??Eu?อณv?&๑w?ay?	๓y?๑มz?Ju{?ผ|?r|?,}?Pq}?๐ศ}?ู~?V~?~?๊ภ~?ง๋~?ฌ?ผ0?L?d?ay?k?
?จ?Iด?oพ?8ว?ิฮ?kี? ??เ?Yไ?่?C๋?๎?s๐?๒?W๔?็๕?A๗?m๘?q๙?R๚?๛?r?ีM>TOฟ#?ฟไ|ฟซ}ฟถฌ(ฟ4uฝ>ต๔v?#๐พ=ฟ๛>v? ๋zฟ;<i??jฟG^{?สxฟม(?เJ=O[ฟ็W?x>่กฟ[ฺx<ษโ?X,8>,งhฟ๕2ฟ๖Rบ>ุท}?R'??ืSพข`ฟuฟํAฟ8ต >	า7?bC|?บ'm?ฃ6?ย;>8พB!ฟลส`ฟ?}ฟ:๋zฟ_ฟพ1ฟl:๓พห๏uพMด๘ปV>z4ฮ>L\?๎4?0VO?ๆฟc?	 r?ไz?๗<?า?Y\}?ษ}x?๏ฤq?ๆชi?.`?ืV?ถL?gB?ใ8?ซๅ-?Sํ#?น@?bํ???ฮ็?>-ญ๎>
K฿>ภะ>Kร>0ถ>ฝ๚ฉ>>ภ็>๚ๆ>ื>็o>:_>๔O>yฒA>7e4> ?'>ฒk>(ฃ>x>ซr?=P ๋=ภฺ=oห=:ฝ=qeฐ==-ค=?ฬ=5=ฏY=Xv=ๅBe=๏[U=?F=Eศ8=๖+=J =P์=x
=๘ =	๐<๋_฿<ค?ฯ<ลpม<8ด<7ง<:ใ<ู<ม?<7?{<๚อi<ฑY<ูwJ<ะฆพGะz?ฆ๏?'จปsพ?ห>@?ิm?ดใพ#๛aฟืZ?๛๛พ.Kพำ>๋ฯพ่A>พKq>AฟSC?ฟ	ฟญ_x?Ui[=q๘ฟฎ๔ผKำ{?3ี>9|7ฟ\rnฟุjพ๑A?Fvz?นs๗>ๆ]พ:b\ฟเำ|ฟ,2ฟR.พฮภ>เAI?ูฐ{?w?ธF?g?๔>?๘>ฐKพL<๛พWo8ฟอDaฟฦxฟ?ฟึQzฟํQjฟbบRฟฃ?5ฟ(ฟ้อ้พ	?ฆพKพ.โฝ๒=:ฉ>I0v>Tจ>D%ั>ฒภ๕>P5?ิท?ๆ&?tๅ1?)฿;?ฆกD?OL?นS?EๅX?^?{b?_f?มi?
ณl?๛Ao?๖zq?is?๑u?v??ฬw?ๅx?ฺืy?`ชz??`{??{?|?แ?|?หe}?๕พ}?4~?O~?~?Lป~?สๆ~?u?-?WI?ะa?w?]?C?ง?๔ฒ?Gฝ?8ฦ?๗อ?ซิ?zฺ?฿??ใ?ข็?ๆ๊?บํ?-๐?L๒?"๔?น๕?๗?K๘?T๙?9๚??๚?]Wr>ลg?บีกฝ๕92ฟอ(TฟBM+ฟ\&ฝ4U?ฐ๗,?ฟYฟ๖cwฝมฟ>?ถyฟi??วฉฟ?&}?๓ฑYฟ[งั>ppฏ>sฎuฟ21?{รๅ>wwฟ>33พฃz?2หญ>JUฟฏไJฟ?r>ปw?Cม<?FUฮฝๅคRฟeูzฟ}ิฟศ=๔)?Kx??้r?C*?ทas>สaXพะธฟ'[ฟ}ๅ{ฟ}ท|ฟcฟซณ7ฟ?ฌ ฟ์พ9ฝ๐๎<>์ร>แฌ?E0?๖mL?ขa?ุp??z??ํ~?Dํ?ิห}?}/y?ไจr?ณj?iทa?
X?๑M?เฅC?S9?/?พ%?8f??
	? u ?ก๐>uแ>ธtา>ีฃฤ>Uกท>bfซ>4๋>f'>6>ฝข>&?q>เ"a>}ผQ>`\C>๒5>ฆo)>"ล> ๅ>ย>ฝก?=	ํ="ฅ?=.aอ=ไ)ฟ=!ํฑ=๊ฅ=^ =ฆq=ไ=O|x=ถ@g=t6W=HH=Tc:=t-=Wk!=ฎ7=?ส==K๒<Qแ<Gญั<Nร<เต<๙จ<4><พS<=+<wn}<iึk<?v[<:L<Dบxฟsmู>3?ล7?mC?ฝ=>?ถ_?โโ?Zบ<ฟฎ?ฟ[?oผ*ฟฯa>ก=
RฝSพ้ณ?ฟiฟp?ฦๅพ๚l8ฟฦd?v>ญ|ฟึ%Sพ๕ฬp?;?$ฟ_ทxฟ~พ%๐,?ฒ~?n{?aLพสJPฟล=ฟ๓q?ฟAtพก>รU??ชx?!8z?๘1N?ๆ?6>ky#พหว๊พ็K2ฟO]ฟvฟ์ืฟู{ฟ้ฒlฟ\โUฟBฅ9ฟผฟ๓?๑พy๛ฎพ?Zพ ปฝ3?ร<้>ฟฑj>mฃ>;vฬ>๔๑>ดW	???%?{0?ฝ:?#ฃC?(pK?ีCR?;X?p]?6๚a?๎e?_i??]l?๖๗n?ซ:q?;1s?ๅt?`v?จw?hลx?oผy?z?CL{?-ํ{?กx|?y๑|?-Z}?ๅด}?|~?G~?~?กต~?แแ~?5?g)?'F?_?t?K?x?zฅ?ฑ?ผ?6ล?อ?๊ำ?าู?๐??_ใ?5็?๊?hํ?ๆ๏?๒?ํ๓?๕?๒๖?)๘?6๙?๚?้๚?0ฟQฃa?ฐH5?Fm=ภmfพHFฝไ+?ด๖?Sี6>บืฟชาำ>z~ม>?ภLฟ?o?A~sฟf๛a?ช%ฟฬQ>}?eโฟ?@??+"?ๆdฟ?ถทพ๏5m?ฅภ๚>ฤS=ฟ_ฟณ2ุ=?m?lO?MUึ;[Cฟ~ฟI&ฟUjปx"?gฟs??คw?๐R5?Ys>฿ป#พHฟmศTฟwฟyฟ$ ~ฟ๒กgฟKง=ฟิฟv?พAฝn:#>Jมท>ํ๋?/l,??rI?โr_?ษo?v4y?Z~?V??N1~?ฏุy?ฌs?Pถk?ฃิb?y8Y?)O?iแD?แ:?R0?J&?t?พ ?ค
?ญu??|๒>?ๆโ>ฒ(ิ>ส>ฦ>$น>ฅัฌ>@ก>ฬf>?=>yบ>#ซs>uc>ฐS>#E>ผ7>โ*>>	'>|๎	>^h >ื๏=(?=ใ$ฯ=ฮภ=สtณ=ง=?s=ฑญ=ฆ=?z=>i=๖Y=,J=a?;=,๓.=dฯ"==>?=6=y5๔<+Bใ<้{ำ<ึอฤ<$ท<์mช<.<ฃ<นW<ธ<ุ?m<N[]<9?M<o^9ฟ ?๑พม4?ึ?'oy?Uื?๛6V?๘ผ๊โ{ฟใฝYi?สmฟฉ?LTดพฌ>(๐พ#นC?๋}ฟัO?>3๖ผ`๙\ฟ฿F?hEๅ>ฬ๓nฟภพ_2_?ถO,?๛พฯ~ฟๆฤพพ?	???เศ%?G?ฝฝฃBฟ?ฟึฃKฟพภ>ีถ4?ู๋t?อด|?VU?์U?ํ`>la๗ฝ&ฺพย๓+ฟว!Yฟหutฟฟ	บ|ฟั๑nฟฌ๏Xฟึ9=ฟ๘ฟiฺ๙พฅ๋ถพ']jพ
Uุฝ0G-<๎๒=2+_>๎ๅ>๖มว>?Yํ>vw?"f?น?#?OO/?๑9?ฬขB?J?xQ?W?tฺ\?ึwa?๛|e?ศ?h?๙l?Sญn?ื๙p?๑๘r?คดt?ฒ5v?รw?ฅx?ศ?y?ขzz?|7{?+?{?i|?๓ใ|?vN}?พช}?ฑ๚}?๓?~?๒{~?๊ฏ~?๎?~?๋?ฐ%?๏B?D\?4r?4?จ?้ฃ?@ฐ?๐บ?1ฤ?5ฬ?&ำ?)ู?^??เโ?วๆ?(๊?ํ?๏?ั๑?ธ๓?]๕?ส๖?๘?๙?๚?า๚?8a{ฟ>>ฃ#?หๆ>?5ณ?>ณ{?ฯ,q?๋.P?ฅAฟพฬbฟcฉJ?ฯFฝtื๗พ็:?J6Gฟh๗,?f์ฟพV฿)พH?ryฟ	s>พI?รถHฟฟ๖pX?n ?)จ ฟฎฟnฟP?แผ๋๎_?T_?`๘่=?1ฟอ๘ฟ6ฟ๏'ฌฝ.q?ลฒm?มT{?'่??แผฐ>fF?ฝ?๓ฟ?Nฟร'wฟ$ฟ9ukฟ?bCฟ๕gฟNOฆพหทฝษj	>5_ฌ>?.(?udF?'1]?๗m?๎Ex?v'~?ภ??ย~?Zyz?B[t?๛ฒl?ี์c?ฒbZ?^P?F?ๅล;? 1?๑v'?mญ?แ8?๓#?u?<c๔>?ณไ>๒?ี>*ูว>5ฆบ><ฎ>ข>๒ฅ>h>า>?ตu>ิ๓d>ทLU>รฏF>ิ9>mT,>สw >ใh>l>ำ>๑=!oเ=่ะ= sย=l?ด=4sจ=Tว=ท้=Fฬ=ฑร|=L<k=v๋Z=ฯปK=l==ปq0=o3$=gฮ=?3=U=งK๖<J3ๅ<Jี<^|ฦ<0ตธ<ฦโซ<'๔<ู<4<|ๆ<F็o<?_<้ฟO<%ซA>81}ฟฮงฝว*??ษ]?IิI?Sดซ>w๛ฟ๊wmฟ*๐>i??rฟ`?<๐.ฟ๕ว ?บ<ฟvUm?๐s|ฟแฅ?;d>'uฟๅว?๚ๆ?เ'YฟXตฟช~G?๏OG?Bลธพ็ฟซ๘พ๒F๚>V~?Cn8?๑?rผD3ฟฟณฐVฟQพพFญB>o)?ชCp?[~?ฏ๘[?"ิ?k>nงฝ๔ศพสh%ฟ[ผTฟHrฟF๘~ฟ6ฏ}ฟTqฟ๑แ[ฟ?น@ฟส!ฟSแ ฟ?ฮพพฏyพ.~๕ฝชฯ4ปูญู=S>ฤง>ร>้>?น?%"?๓ .?าr8?ฃ?A?rญI?ฃนP?ษโV?ฐC\?o๔`?
e?8h?eฑk?bn?zธp?0ภr?at?๓
v?ฎ^w?[x?ๆy?vbz?"{?ษ{?IY|?Pึ|?ฅB}??}?ำ๑}?D8~?Ju~?&ช~?๏ื~??~?๑!?ฑ??uY?ลo??ี?Tข?แฎ?ภน?*ร?Rห?aา?~ุ?ส??`โ?Xๆ?ศ้?ร์?W๏?๑?๓?.๕?ก๖?ใ๗?๙๘?๊๙?ป๚?ฎ(พพv4ฟT???๎ทt?<[z? t?Tญ>TOฟขาฟฃK|?#??พฏซฝเำ>ิจ ฟ@Vศ>#,ฝ?บใพฉึk?+cฟฦa=~>g?9$ฟY.ฟkุ<?ฮ??./ ฟฦ~yฟ$พ/?N?้Bl?ฎํ`><>ฟ7ฟcWEฟ๏'พ๎๕>๏}f?จ๕}?สI?`wห>๎๓dฝ๊-๐พ๒Gฟtฟฤฟ"๘nฟซไHฟขฟI|ดพาS๏ฝ฿=Oไ?>o?>?$?พCC??Z?~๒k?SHw?6ฐ}?๗?.?~?w{?)u?ฉm?๛?d?บ[? Q?๖OG?๛<?หท2?ณก(?ฯ?๚O?i/?นt?`H๖>\ๆ>wื>๒rษ>๎'ผ>งฏ>ป่ฃ>ืไ>ต>t้>Rภw>๛?f>W>@YH>ฮ:>ญฦ->ั!>ฎช>OF>?> #๓=Tโ=.ฌา=ฑฤ=ถ=ั฿ฉ=ษ=ป%=s๒=[็~=:m=๒ล\=puM=u4?=I๐1=y%=ร=h=t=ำa๘<h$็<+ื<ๅ*ศ<ืEบ<Wญ< Oก<l<ฏฐ<?<ด๏q<๊#a<Q<Lฐm?๊5ฟd๏Oฟjะฝ_Q>FิU><๋พSโpฟ๊ฟ?X?	->gQ^ฟ!?ํiฟR]?ืkฟh?~Heฟ )ว>๐?์>๓ฟ?ง?>ฺ}D?t;ฟ?ึ,ฟW*?]?^eพฑ|ฟสัฟ1%ล>nฟy?็;I?ฬ?=\#ฟซ|ฟค`ฟฎห?พK>ฺ?๔๊j??ษb?!?๕#>&s.ฝo?ทพ์ฌฟ| Pฟoฟ๛>~ฟ7z~ฟ$sฟอธ^ฟV%Dฟ%ฟหฟ๘คฦพgyพ5M	พ{ึผ}dภ=ขH>e>4Jพ>๛?ไ>6ฏ?p
? จ ?kฐ,?ตJ7?ช@?คษH?W๒O?ญ4V?ฤซ[?p`?;d??4h?Zk?:n?vp?๘r?ถQt?ฺ฿u?K9w?๎dx?ศhy?Jz?f{?ฑถ{?jI|?ศ|?ป6}?/}?แ่}?0~?n~?Vค~?ๆา~?;๛~?*?k<?V?Qm?๙???ผ??ญ?ธ?!ย?lส?ั?าื?5??฿แ?่ๅ?h้?o์?๏?T๑?K๓??๔?y๖?ภ๗??๘?ะ๙?ค๚?๏ข?ek}ฟ๓vพ"?Mf?.b??>พั|ฟ๒AฝFs?sSฟตำฉ>ึฆ	=$พ$?~=E{>๔x/ฟ	๛}?ก>ฟ~9พ'ญy??(๑พ9วNฟK??X?![นพ ฟKพ:?จv?pcฅ>ุ\	ฟํ@|ฟจดRฟ,ดxพยำ>เ)^??M๏R?3ๅ>ฉkปUโูพm?ฟ>จpฟก?ฟx)rฟ+Nฟ&ฟAยพ๔vพซ=@R>ก๒>
r ?ฦ@?*xX?{Ej?ต;v??+}?ใ?%?ก{?ผ๐u?n?f?ช\?ธผR?๑H?ไ.>?็3?แส)?๏?f?:?7s?e,๘>/J่>A@ู>"ห>,ฉฝ> ฑ><ฅ>|#>!ฝ>ฒ >สy>์รh>??X>J>ฌ&<>ึ8/>'*#>j์>'r>กฎ>ฌ+๕=๎8ไ=ฤoิ=:ผล=ธ=iLซ=9n=ปa===ิ7o=l?^=/O=}ฯ@=ึn3=๛&=e=b==?w๚<้<ส็ุ<kูษ<~ึป<xฬฎ<ชข<P_<*?<ผ<"๘s<8c<IES<*nP?่>xxฟัถEฟง฿พ~ะ๏พ=Qฟ:wwฟdฝขี?ulพ%Oฟิq?๛ฺฟ:ฌ|?@ฟg'x?ถf:ฟ{] >๋+?ฬร{ฟs1b>๖าa?'ํฟุKฟ"?กขn??จฝl?tฟrC/ฟ+c>ฉFr?ะX?r.>dฟณUxฟiฟie?พFX{=w?ิd???ชฉg?S?)?&ฎ>ฆH\ปฆพ'ยฟUOKฟะฮlฟโT}ฟ๋ฟ๗?tฟๆsaฟ{Gฟ๗@)ฟ๑ฉฟmฮพ?พิพ ๑ผง=ik<>>๕น>bเ>Fว?้X?Y(?ฝ]+? 6?ไ??5ไG?)O?EU?ฑ[?๊_? #d?ตฯg?%k?รษm?'4p?HMr?ฃt?fดu?w?<Dx?nLy?1z?๘z?:ค{?i9|?ฑบ|?ท*}?ฦ}??฿}?ต(~?ะg~?y~?าอ~?ี๖~?[?9?ฤS?ืj?ิ~?#? ?ฌ?Zท?ม?ษ?าะ?%ื???\แ?xๅ?้?์?ล๎?๑?๓?ะ๔?O๖?๗?ผ๘?ต๙?๚?[ฒ?ฝ๔พ3eฟ'๐ฝ๋ฉธ> ภ>ฯฝ๊Bฟะฦ`ฟย๏>ฑ1?I}ฟdS0?าสณพร8O>Rพq๒?y?]ฟ2}?sฟฟpะพ?????พ(ฎgฟ้ฏ้>Jl?ม้[พUฟ~Wิพ3x#?a|?gุ>18ๆพ}wฟญ}^ฟฤํฃพฐ>๘ภT???tO[?=้?>G=รพ=ฃ7ฟbรlฟาีฟ%uฟ4Sฟุ๊!ฟ[ะพฐ'/พn=ฒช>r่>ลP?ูห<?JV?h?% u?ป|?ฤ??b?๘'|?ฐv?o?g?ศ]?ฯๆS?	ณI?ื_??U5?y๒*?ค!?{?ฦC?q?H๚>๊>L๑ฺ>ธคฬ>๎)ฟ>ืzฒ>-ฆ>฿a>Y็>ล>oิ{>ฅซj>ภฃZ>อซK>mณ=>่ช0>9$>.>๒>๙ล>)4๗=ภๆ=O3ึ=ป`ว=,น=?ธฌ=ฅม?=ท=ฤ>=P=5q=โz`=ซ่P=jB=`ํ4=_(=vฐ=มะ=ฒ=(?<ก๋<jถฺ<๑ห<$gฝ<QAฐ<ค<4ข<ฅ	<[-< v<์d<๘U<~SGฝCไ`?Pไพ?;~ฟ๗ฤnฟT6mฟืฟde&ฟ๊
๕>i2b?I8ฟีญพ"9?sฒoฟดz?น2vฟSJX?ษ??พ๒'พ1๋U?6ำiฟฒ;ยชu?9ฯูพวcฟU?ล>?z?คqx= ๒hฟEฟ~'>Zh?(ฉd?jฦ>ห??พrฟ๋bpฟaฟํฝgป?^?1ฒ?๏ฐl?ำ\2?฿ธย>~๏=Cพชฟ$JFฟุiฟ(:|ฟ7ฟvฟใdฟ&ผJฟไ,ฟะ}ฟ&ึพคพS&พF/ฝผ=ยศ0>ะา>๕พด>ฮM?>ฑน?>๗ค?Gฆ?์*?๔4?T>?(?F?a_N?ิT?vxZ?d_?เญc?รig?{ฉj?ฐ|m?1๑o?"r?(ํs?u?ํv?F#x?ู/y?นz?โz?{?F)|?ตฌ|?}?F}?ฤึ}?ิ ~??`~?~?ดศ~?f๒~??ห5?โP?Yh?ซ|?D??ดช?#ถ?	ภ?ศ?ะ?vึ???ูเ?ๅ?ค่?ล๋?{๎?ิ๐??๒??๔?&๖?x๗?๘?๙?v๚?UW?>;ึ>\kฟ<2LฟWพพWSพอ้!ฟ?ข~ฟXญ๙พ;เX?>๎?sฟ"ุl?บ.ฟ,๋	?ๅ5ฟXkE?nyฟธi?OPคพๆ;ฟ่y?L ฃฝั%xฟลj>v*y?mฃฝ"ฑzฟ:3ฟลำ	?ๅฑ?Kv?ิiทพฅาoฟุhฟqสพ๛>๒NJ?kf?ฅโb?}ป?o?ฮ=Hฬซพ^/ฟยrhฟ*Gฟ/wฟข?Wฟ(ฟว?พฬถJพห=ง?{>Fj?>?Gu9?yS?Oบf?ณ๕s?๚{?ๆ??Pฆ|? iw?ัhp??h?rแ^?aU?9เJ?g@?-A6?x,?s,"?๕?ฉL?n?๑๛>?๋>ก?>ณ<ฮ>2ชภ>(ไณ>lใง> ?>[>ญ.>?}>&l>k\>?TM>@?>โ2>9?%>ดo>ฐษ>G?><๙=่=ะ๖ื=3ษ=ดป=%ฎ=ข=ฐู=่d=ฉ=M3s=UUb=DขR=D=๊k6=ร)=ฮ๛= =ั=Qค?<ผ๗์<?<w6อ<ส๗พ<)ถฑ<
`ฅ<ๅ< 6<๛D<?x<ิะf<งสV<sไ]ฟลnh?ถgษ>ูfฟ็ฆmฟcsฟ?JFฟ?ำฝz_??ณvฟ๒>ีSย>1;ฟฌW?มปPฟ4๚"?.fพึดัพqur?<ฬJฟ^พว/?Oฌ{พ:?tฟk>=??qO>๏ยXฟปWฟ2๊G=ิ[?o?ยณ>ะีพ#kฟrNvฟ ?ฟ1 ฝE$ํ>^V?ฑ~?(q?:?๖ึ>ฮด=uEพ#h	ฟ0Aฟjญfฟ๏zฟ?ฟ?"xฟrfฟ็ๆMฟx0ฟOFฟ(ั?พ.,พ2ษ4พชeฝฮฝh= %>>V๒ฏ>X?ื>ํ฿๛>ข๎?๐!??ฑ(?ฦ3??=?F?บM?"T??Y??^?ู7c?g?Pj?/m?ณญo?ุq?Eบs?n\u?Gวv?x?y?ฝ?y?๓ฬz?ึ~{?|?|?b}?ฑv}?อ}?ใ~?Z~?~?ร~?ํํ~?ค?p2?๚M?ีe?}z?a?฿?Jฉ?้ด?๙พ?ฐว?<ฯ?ลี?n??Uเ?ไ?@่?o๋?1๎?๐?ฅ๒?o๔??๕?T๗?}๘?๙?^๚?xx๋พ~z?ุพWห|ฟ๎gฟฯXฟ?xฟ็`ฟยX=}ำ?ฝ3Rพms9ฟ????hฟ๕ืM?FOQฟIPn?๔kฟ๔{D?ฝฯGฟฑg?K >ปฟ?ผ๑=เp?6ป=Kทpฟฯ#ฟd!?>|n?1?Qฦพvfฟพ๛pฟค๏พ5L>าเ>?ขน}?อกi?D?ยu>ศพคฝ&ฟSธcฟใS~ฟถษyฟึ\ฟP .ฟ;๋พ?fพไ๋;งCd>`/ิ>??^6?ฤ฿P?f?d?rผr?"N{?b?Jฟ?	}?]x?๒Fq?i?}๖_?g0V?~
L?บA?k7??<-?๓H#?าก?ฎT?}j?ขั?>$ฅํ>#Q?>ิฯ>๘)ย>Mต>Y6ฉ>??>';>jE>v็>nzn>92^>ว?N>ฬ@>ฤ3>$5'>Aฑ>b๕>๔>๑D๛=?็้=Gบู=ฃฉส=4ขผ=ฏ=ohฃ=ฅ=	=๊บ=1u=ฦ/d=?[T=?E=q๊7='+=%G=~9=๐	==] =ื่๎<ฆS?<?ไฮ<oภ<+ณ<ปฆ<?'<b<\<kz<!ตh<WX<๎Qcฟู:S>t?~!>5ฺพ!ฟpพํ๕>+ค?๙๏=\zฟ?y0?กฆฝิพุ2?Geฟป>ฏ=$ฟษV?sk ฟดภูพ๛}??Bjฝ?5~ฟ6G=ํ~?Cฎ>lบDฟ!gฟณfฝRK?ธ๘v?M๊?>
วฌพZ:bฟืzฟv*ฟ5พ่Mั>ON?~?|?ษu?L@B?j๋้>Fต้=บ6`พ!?ฟะจ;ฟฉOcฟฒsyฟO?ฟ?yฟC๛hฟn๛Pฟ๛3ฟ ฟ,lๅพเชขพ?5Cพถฝf๚5=พq>บ`|>5!ซ>ชำ>T๘>๐5?Z?๒X'?2?฿z<?;*E?ขฦL?GoS?@Y?T^?ํภb?f?๖i?ทเl?ฌio?pq?๛s?๋/u?จ?v?เw?๛๕x?ๆy?ทz?้k{?|?f|?}?l}?Zฤ}?แ~?+S~?~?Vพ~?k้~?ผ?/?K?Kc?Jx?z?9??ง?ฌณ?็ฝ?รฦ?nฮ?ี?ิฺ?ฯ฿? ไ??็?๋?ๆํ?S๐?m๒??๔?า๕?/๗?^๘?d๙?F๚?Xๅ~ฟ???(	?ํฟ)ฮsฟ2}ฟmjฟโฮๅพ?ย?๚Mb?xฏ#ฟึ๐ฒพฐf?yุฟ^v?Exvฟ?4toฟผศ?๗L:>gฟW>J?aฆ>ต}ฟTขsฝฃผ~?:่x>*ฮaฟT<ฟ8ก>ฤ{?"0?$)พแ[ฟOwฟhค	ฟ)ฦ >ึ2?๛z?o?`์"?ค-Q>ล*xพลฟ>^ฟ\?|ฟ๚ช{ฟูิ`ฟ๔๘3ฟzว๘พญพ๊?ผหL>ฺษ>?q2?5N?q๎b?utq?z?ฃ?a??}?Dฤx?ๆr?j?<a??OW?ำ1M?PไB?๓8?_.?d$?ณ?ำ[?*f?ฑ?>Bl๏>์?฿>ิjั>?ฉร>ตถ>๒ช>y>ฝd>๛[>H๘>}ap>1๙_>ฆP>?XB> 5>?(>ฟ๒>!>ฤ>=M?=๊ห๋=ฒ}?=
Nฬ=ฎ)พ=?ฐ=ฮปค=Q='ฑ=ดฬ=ท.w=3
f=qV=;G=๗h9=,={ =?m==Ph=๐ู๐<C"เ<ะ<ย<ูด<๚จ<฿j<<9t<ุ|<nj<PZ<sพฝHW)ฟ(X?EVR??#>>\อ>ขรd?@Q?฿Y๏พ'ีDฟ7?o?รเพ =ฝฏN>efพาl=lต>เSฟ@บ{?' ฺพi๓ฟ9r?ญ>๗ฟ้1หฝpRx?ำ:๑>ง3-ฟ/๚rฟฏx9พฦ9?w|?d?rพ>ูWฟน๗}ฟW|7ฟ$ศIพ๎ณด>มvE?z?J^x?์I?d้?>ะฌ>;พdื๔พg6ฟฟ_ฟศwฟ๕ฟlืzฟ	Dkฟf๙Sฟ(m7ฟ๗ณฟ๗์พณชพFQพ๊Qฉฝา/=-พ>๎ฒq>ณKฆ>'Qฯ>๘๔>๊z
??ั?%?ธd1??m;?_>D?๘K?ถบR?ๆขX?ส]?Ib?23f?Ri?าl?%o?ไaq?ISs?u?ปyv?วพw?ฒุx?)อy?กz?ีX{?๘{?|?ง๙|?Da}?	ป}?ฮ~?-L~?~?น~?เไ~?อ
?ฆ+?H?ผ`?v???lฆ?mฒ?ำผ?ำล?อ?`ิ?9ฺ?I฿?ซใ?w็?ม๊?ํ?๐?4๒?๔?ง๕?
๗?>๘?H๙?/๚?ิดฟ้์จฝ7}?QพJ>e๚พั+ฟ/^๙พbณ>wl?u6?๒รlฟฤ=๊๐#?2ฬoฟฐเ~??pฟ|}w?DๆJฟยค>ซู>ึพzฟ?"?lฯ?์rฟะรpพฎw?(ว>DNฟ&?Qฟ๋ฉG>ฟt?EC?ฦำฝื?Mฟ๐C|ฟฟG7Q=eJ%?-w?t??ร-?uฉ>XGพจyฟุYฟA{ฟV6}ฟg?dฟฃ9ฟJ๋ฟา2พ2B3ฝ~ฎ4>kฟ>'?า
/?zK?๐`?ฯp?ฒอy?ั~?b๓?ํ}?ะfy?จ๐r?pk?จb?มkX?4VN?กD?ฺธ9?ย/?๘}%?Pฤ?b?a	?ฏว ?p2๑>๑ญแ>๗ ำ>(ล>ณธ>7?ซ>ะX?>>_r>ฑ?>RHr>๛ฟa>.OR>FๅC>?r6>มๆ)>-4>?L>๔"	>wU?=ฐํ=A?=i๒อ=!ฑฟ=kฒ=(ฆ==Bื={?=f,y=ไg=ฯW=ึH=|็:=๏-=ะ?!=7ข=.=cs=	ห๒<เ๐แ<Bา<นฉร<ฐถ<๒pฉ<ยญ<ป<ุ<E"~<ป}l<ด\<ฉงI?ฏ ฟ:J>ก๎z?L_?ทฦ=?_?}??'ฤ>่ภXฟ!ถยพpั~?จDฟJAณ>ฟฝํG=๔้พ?irฟ?g?๐gNพษiEฟ[\?axฃ>มาxฟtพมึk?J??ฟ!{ฟว"พะ%?ตo?-?ถE.พLฟtชฟ|Cฟu@พq>ง?;?w?{?P? บ?;AD>Bตพ(lๅพdG0ฟษ?[ฟปํuฟ5มฟภ๛{ฟzomฟ{เVฟ&ฮ:ฟXฟlq๔พ1ฑพฉ๏_พjฤฝ"ภ?<ฦ>??f>๏qก>๓ส>์5๐>ฝ???$?ํ00?]_:?์PC?%(K??R?X?@]?hะa?สe?โ?i?RBl?เn?ใ%q?/s?ีีt?Rv?พw?/ปx?ณy?์z?E{?g็{?กs|?#ํ|?lV}?คฑ}?ซ ~? E~?n~?ฬณ~?Kเ~?ี?6(?E?(^?ุs???ใ?๙ค?,ฑ?ผป?โฤ?ฯฬ?ซำ?ู?ม??6ใ?็?i๊?Nํ?ฯ๏?๛๑??๓?|๕?ๅ๖?๘?,๙?๚?๔จ>u7YฟแY??K X?ึMr>^ฝ>?4-?pX|?๗=฿ฬ~ฟ๏ฆ?ญ3>GK;ฟ7e?h2kฟว?V?ฏๆฟฅภZ=x#?บูฟzค็>๏,?,ฉ^ฟฮพจh?0?[6ฟ?~cฟAก=wj?๙ณS?Rธ	=ฤำ>ฟฟ๚~*ฟCฎรผ๚A?cUr??ซx?>8?`>ิbพ)เ
ฟฆ$Sฟี"yฟCk~ฟN?hฟ?ฟัU	ฟึพัปฝ.ป>%ๅด>vถ?ึp+?9ฏH?๓โ^?ธn?ย๙x?์w~?K??NI~??z?1ผs?๖k?พc?Y?wO?0E?ภ?:?A?0?y&?์ำ?ug?[[
?=ถ?ฉ๗๒>0[ใ>{ิ>Kฆฦ>gน>(-ญ>ใก>Eท>>๖ >๎.t>c>จ๗S>qqE>ุใ7>r?+>u>,x>:
>ัฎ >๏=i฿=พฯ=8ม=ืณ=}bง=oษ=[?=?๐=*{=ฟi=Y=qJ=?e<=S/=$)#=ึ=๛L=u~=!ผ๔<{ฟใ<๐ำ<]:ล<ท<้หช<ฅ๐<่<wฃ<Y<bn<cี]<๒จq?yฟ?*ฟฝ4	?ูบx?ึะ?x}?<?Dh,พKัฟฑ
ฦ=}WZ?bvฟ็.?ฉลโพ",ส>:-ฟษ=P?yขฟ9E?ย๗=Mdฟ;ย<?ณฆ?>@jฟ๚ีพ/ฦY?n3?-ส๊พUฟ?อพุ๏?๒ฺ?ฆ*?ีฑฌฝฑ๖>ฟMํฟ5Nฟ
ฅพcAs>e๎1?vฤs?f9}?W?ฦม?Kk>!BใฝฝีพAR*ฟHXฟยใsฟ฿bฟฮ๛|ฟS}oฟ_ฐYฟญ>ฟ๐ฟฺ๛พ่้ธพG;nพ๗ฏ฿ฝ+i์;?์=6B\>	>ฦ>CI์>???S๚?`A#?:๛.??N9?ไaB?รVJ?ฝMQ?'dW?ด\?ฯVa?9`e?ยใh?7๒k?jn?j้p?ฎ๊r?Cจt?๔*v?qzw?ox?วy?tz?82{?ึ{?e|?เ|?~K}?,จ}?w๘}?>~?Fz~?wฎ~?ฌ?~?ี?ฟ$?B?[?q?ญ?3?ฃ?่ฏ?คบ?๏ร??ห?๕า??ุ?8??ฟโ?ซๆ?๊?ํ?๏?ม๑?ช๓?Q๕?ภ๖??๗?๙??๙?I?x?ognฟ๗ฑผฆx?ฺU?๏?f?5?ปฃy?>?ฏ๑๎พ?UฟมW?bพ2ขิพๅ-?<ฟGํ ?ั3คพเ_พQฺO?ขขvฟx)|>jสO?7Bฟแฟ8ใS?พb&?๔?ฟณqฟ%ญTฝo]?o๋a?๛m>๖$.ฟุ?ฟP9ฟ`,สฝ}?ญwl?ไ{?ๆรA?๚ฉต>ฮษฝฌ? ฟXฺLฟYขvฟYIฟrlฟธhDฟ็กฟ๛่จพ๙พมฝOฑ>9Hช>เ6?ำฦ'?ำE?ฑล\??Dm?ปx?ฎ~???a~?ฦz?}t?zเl?wd?ทZ?P?๋RF?ข?;?พ1?กญ'?oโ?๏k??T?4ค?๎ป๔>ชๅ>]+ึ>$ศ>ฒ์บ>ฤ~ฎ>ฑาข>6เ>ฃ>>Ov>Me>?U>}?F>XU9>,>ูถ >ซฃ>3Q>?ฒ>y๑=ดวเ=;ั=๑ฟย=๔Cต=อตจ=U=p#==น'}=hk= B[=yL=ไ==ท0=wt$=ํ
=๔k==8ญ๖<ๅ<
ี< หฦ<^?ธ<เ&ฌ<3?<<ป<<TFp<_<๏m>?บ>๐ฟasพ|น?ZAJ?,4?ืb>ึ+ฟ~ibฟb?Xส	?เ/}ฟแh?ภ฿;ฟpณ-?%GฟEzr?ฮyฟiq?77>เxฟ?'Q&?็JTฟCชฟ.B?eภK?๒5ฌพงฟjฟอ๐>ฝท}?ค;?yb;0ฟไฟ~ฟwXฟ'ฤพHฝ6>#N'?Xo?ม~?? ]??B9>,ฬฝ?ฮลพ1$ฟโ่Sฟ๘ชqฟฺ~ฟqื}ฟPmqฟยh\ฟo[Aฟ{"ฟฟf>ภพ_z|พUะ๚ฝ7ชปี=ํQ> ฒ>%+ย>X่>&<?๘j?เ!?ฅร-?โ<8?IqA?๖I?XP?รV?(\?S?`?๕d?๒h?กk?ETn?{ฌp?ฦตr?Vzt?v?เWw?tx?ศy?^z?ฏ{?ฌล{?fV|?ฯำ|?z@}?ก}?2๐}??6~?t~?ฉ~?ื~?ฬ?~?A!???๑X?So?ถ??
ข?กฎ?น?๚ย?(ห?=า?_ุ?ฏ??Hโ?Dๆ?ท้?ณ์?J๏?๑?x๓?&๕?๖??๗?๔๘?ๆ๙?ํ8?ฦ;พฦFฟฟA ?f|?้5z?kห}??ฆm?	?>กXฟfx๕พชM~?)
ฟำฝ*ฌพ>อ๎พถ>~โฝแ๒พ่ฯn?_ฟtyๆ<k๊i?r{ฟ?\2ฟาS9?5BB?U๘พdzฟ?U3พทL?ชm?้m>a๔ฟ๖?~ฟB๎Fฟ$%1พ,๒>วe?0~?ผ?J?tฮ>?ทLฝฤฎํพฬ2FฟฆภsฟJะฟสXoฟIฟ&ฮฟณถพํก๕ฝซ(ู=>wR๛>!$?ํ็B?๕Z?ภยk?ช*w??ก}?ุ๕?ฤๆ~?&"{?@u?!ลm?ะe?ภฉ[?sฑQ??rG?|=?Hฺ2?mร(?ื๏?o?ฃM??;๖>Zณๆ>ฟื>Nกษ>Sผ>
ะฏ>:ค>๎>ด>	>u๛w>=g>+HW>jH>ฟฦ:>๐->๘!>ฯ>Ch>โถ>^๓=๔โ=O฿า=OGฤ=_ฐถ=	ช=8A=I=ม=\%=ษsm=ซ๛\=pงM=?b?=2=ษฟ%=G?=์==N๘<ฑ\็<Mื<ฃ[ศ<4sบ<ึญ<ivก<๛@<ตา<ล<ก*r<ภZa<a1ฟช่v?ฆ!ฟี]ฟ+พีX>`> Yพพิ>vฟShฟ๑จ`?e๋=Wฟโี?๕mฟsb?Doฟ??ๅ\aฟLqธ>ZA๙>ๆฟ|ะ>฿@H?5ฃ7ฟ00ฟjป&?ฬ_?ั%Uพ#|ฟ๚ตฟX๓พ>y?AK?sต=!!ฟ๊#|ฟืaฟKnโพห๓=y&?๋Cj?ฎ?hพb?j"?Gx>Dฝ)ฆตพภๆฟOฟษCoฟ์&~ฟ~ฟ6?sฟ[	_ฟDฟ๙%ฟฟ;ฟ8วพVพค๒
พ_4ผฝ=sทF>Vฬ>fภฝ>jbไ>x?zู?ส| ?/,?)7?@?ฟฏH?ฏ?O?โ V?~[?๓``?d?r)h?1Pk?n?op?vr?Lt?๓ฺu?
5w?>ax?ey?PGz??
{?ด{?G|??ฦ|?`5}?}??็}?ฃ/~?ัm~?ญฃ~?Tา~?ผ๚~?ผ?<?MV?
m?ป?ศ???Xญ?lธ?ย?Rส?ั?ฟื?$??ะแ??ๅ?]้?e์?๏?L๑?E๓?๚๔?t๖?ผ๗?ื๘?อ๙?ไFพo?aป{ฟพึ?b?i^?ป|?,&พ
ฯฟํญUผนq?EVฟณทฒ>.<`พ้ยA=ฆ>*?1ฟแX~?y<ฟอ๐Cพ6z?ํพตPฟ+ด?ถY?"ถพ)Aฟ/_พ}ค9?_fv?ปชง>Qjฟ?|ฟlASฟ15|พา>ฦ]?ซ?cQS?ญๆ>ล๒ชบBๅุพ	1?ฟ?~pฟ้?ฟbKrฟฺdNฟ0ูฟwรพฎพvฯจ=ะ>๒>D ?์??็\X?Q2j?/v?%}?|โ?s(?ง{?I๙u?คn?ยf?#ท\?ูษR?QH?L<>?ห๔3??ื)?!??.r?ญE?X~?A๘>B^่>;Sู>
ห>บฝ>๚ ฑ>}Kฅ>o1>2ส>๋>`แy>Iูh>2๐X>8J>8<>I/>D9#>๚>H>?บ>xB๕=)Nไ=ิ=ฅฮล=ฤธ=a\ซ=}=o=%=~='No=3ต^=fBO=}แ@=3='=?s=ใฉ=ฉ=c๚<J+้<?ุ<F์ษ<
่ป<ฬ?ฎ<Kนข<tm<S๊<๛!<ํt<nc<ป){ฟ*นI?4:>๔uฟ)TKฟn๎พP?พYUฟฅuฟ๎Kฝm๚??รฉพฒฟ?ๅo?(๗ฟ`L}?ขถฟXBw??8ฟ}X่=ฌ-?
E{ฟ X>9ๆb?ฮ0ฟ?ธLฟx๖?=Ao?ฏ-ฝWtฟaF0ฟกๅ>Yโq?H?X?ฟง2>ลฟ!xฟviฟนภ?พฃทo=n?d?๒??ๅg?=*?yฏ>$QืบรGฅพsฟ!KฟญฎlฟI}ฟ? ฟศ๒tฟ฿aฟn?Gฟj)ฟีฟํรฮพ?gพสvพ.ี๕ผ๗ฅ=้;>สโ>tQน>`hเ>ูฑ??E?}??N+?6?c?? ฺG?ร O?}U?๖[?ฒไ_?ไd?Bหg?G?j?fฦm?;1p?ฟJr?ot?}ฒu?๐w?ฬBx?/Ky?l0z?(๗z?jฃ{?ต8|?บ|?/*}?P}?w฿}?](~?g~?7~?อ~?ค๖~?0?๙8?คS?ผj?ผ~???ฌ?Lท?
ม?zษ?ษะ?ื???Wแ?sๅ?้?์?ย๎?๑?๓?อ๔?N๖?๗?ป๘?ด๙?lnฟ๎?mG?พbฟ๘ฦฝฯม>ภ๚ศ>ขLผ๒?ฟ๗bฟ๓ช้>ใ๚3?ึ|ฟ(O.?๕๕ฎพHำE>ฐพq0?ฆ]ฟ`t}?kฟอพ??พ฿$gฟ๋ย๋><ฉk???_พฟvฟาพm$?]m|?" ื>d็พ(Dwฟ5^ฟ+ํขพ?ฐ>U?ใ??[?OE?>B=ฟฉรพ@ุ7ฟK?lฟ&ุฟZ๖tฟLSฟฎม!ฟว ะพ^r.พ็ปp=o๗>~ห่>l?แ<?ญV?ซh?'u?ฃ|?	ล?la?$|?ภซv?~o?Jg??ภ]?6฿S?FซI?X??5?๊๊*?L!?๏s?๘<?j?๋๚>^๊>4ๆฺ>@ฬ> ฟ>qฒ>zฆ>ถY>ด฿>>ว{>%j>Z>็?K>Bฉ=>lก0>az$>ุ%>C>ะพ>ั&๗=Sๆ=ผ'ึ=๔Uว=#น=ฃฏฌ=๒ธ?==97=L=(q=ธn`=Y?P=๙_B=ใ4=iV(=๗ง=ฺศ=นช=w?<ใ๙๊<ชฺ<่|ห<฿\ฝ<ย7ฐ<-?ฃ<ํ<๑<0&<9๓u<เd<ผพ6หผผ้]?ผ/๎พนฬ~ฟR๓lฟอskฟ๔ฟยc)ฟr๎>?ศc?~6ฟ$p พN};?ุpฟh-{?หฬvฟณdY?**ฟ๔
พ??T?Jtjฟถ?;?Hu?ข?พฃ>cฟข?ว>ฬฮy?cli=ฐNiฟฏ}Dฟd{*>oNh?l]d?><?พTฑrฟ{7pฟ?ฟ
K๕บpe?3^?jถ?ธl?.&2?อ5ย>Uั=ธพูฟนkFฟ"์iฟB|ฟทฟาvฟ
dฟงJฟ่ฬ,ฟีdฟ๔ีพrพ ๔%พ&ถ-ฝfb=11>๕>e?ด>j?>ๆา?>+ฐ?4ฐ?ฏ*?G?4?>?G?dN?ูT?n|Z?g_?ใฐc?clg?รซj?ซ~m?๊๒o?กr?u๎s?ธu?๎v?$x?0y?\z?*ใz?{?ฑ)|?ญ|?้}?}? ื}?!~?*a~?ท~?ีศ~?๒~??แ5?๕P?ih?น|?P??พช?+ถ?ภ?กศ?ะ?zึ????เ?	ๅ?ฆ่?ว๋?}๎?ึ๐??๒?ก๔?'๖?y๗?๘?๙?=ลOฟ๓'?ซฌ>Nุrฟ๊ว?ฟ+พ๕\yพ#_ฟuณ|ฟ+ฟุQ?8ค>wฟยh?ก'ฟขY?2dฟxn@?{ธwฟ&1l?้ฐพ~ฟ0{?๛๑สฝภ๖vฟฌ>๔5x??นฝ`{ฟ?pฟฬ\?ื?ฃ7? ๓ปพpฟธgฟNีฦพ>4XK???Q3b?n
?๗|ฤ=eฎพศ+0ฟYเhฟYฟ๊XwฟตWฟQ'ฟ)ผ?พถHพ;ถ=H~>ฯa฿>s?Dว9?rทS?ๆๆf?ถt?A
|??ฎ?ญ|?ๆWw?aSp?bh?ใฦ^?๑T?ธรJ?พq@?ย$6??+?V"?ลt?3?V?Iร๛>ฎฑ๋>x?>๑ฮ>ฐภ>ำมณ>0รง>ฤ>	๕>,>ฌ}>ะdl>ฬ?\>v,M>]?>ท๙1>nป%>"Q>2ญ>บย>๙=qิ็=ๅหื=;?ศ=|๕บ=แฎ=ษ๔ก=งป=๒H==ุs=;(b=JxR=s?C=G6=ทก)=N?=ะ็=ษต=q?<{ศ์<Y?<อ<ดัพ<ธฑ<?ฅ<fฦ<<f*<ืw<สขf<ถ?จลPฟจ q?:ข>)ฟปำsฟKxฟs็Oฟ?#พX?Cข?xyrฟVษ>ม'ี>!|Aฟ6T\?YiUฟiึ(?ข(พ>zลพศep?DNฟIพoฝ~?ผิพ๚sฟ1ญz>[=?ลรA>xxZฟ๒Vฟะt=Oh\?ญ n?ภ?ฎ>ฃูพP่kฟอuฟฟ*ฝพ๏>ฯ>W?ฎั~?ยp?ปฝ9?mฅิ>:=J?พ
ฟUAฟฏ?fฟ{ฟฆืฟ ?wฟYfฟศMฟบ!0ฟฝ๊ฟ@?พIuพงi3พ?z`ฝฝm=<&>๋>Tgฐ>sgุ>ู=?>g?๔F?ญา(?Xใ3?I=?ฏ*F?&งM?3T?็๋Y?้^?Cc?ิg?ฅXj?j6m?#ดo??q? ฟs?ฅ`u?๏สv?6x?วy?z?ฯz??{?|?๕|?}?ดw}?yฮ}?ฅ~?รZ~?,~?ฤ~?Z๎~??ย2?AN?f?ฒz???mฉ?ต?ฟ?ฦว?Oฯ?ึี?}??aเ?ไ?J่?w๋?8๎?๐?ช๒?t๔? ๖?W๗?๘?๙?-nY=Nุพ~o?v?พ๊ฟ๋pXฟหGฟฝqฟ	กkฟ?๙!ฝd~?0VพFฟ5ำ?ฌaฟ๋UD?~ฟHฟ<!i?d๔ฟขฌK?๕่ฝtAฟ~k?ํัป=?~ฟ|>ิโ~?ม=็ณrฟโtฟๅ>ต?!?แพ\$hฟyบoฟV้พแX>แา@?
~?h?ง?Rพ>	พ /(ฟdฟฮ~ฟ]ryฟ๒ฯ[ฟั%-ฟ,M้พหaพyo:<(h>เี>?๘6?_NQ?,e??๐r?gk{?ๆk?5น?D	}?ถ?w?ี"q?๐h?6ษ_?ว V?ขูK?[A?1:7?฿-?>#?ฏt?P)?A?จ?>0Zํ>4
?>ฯ>เ๊ม>ผต>?จ>ฉ>/
>>ธ>J*n>^็]>ๅทN>^@>๎Q3>i?&>^|>ฤ>ฦ>Z๏๚=้=pู={dส=ฯaผ=Vฏ=0ฃ=ฌแ=งZ=แ=,?t=ผแc=9T=์\E=ซ7=ํ*=ฅ=ล=ุภ	=N1 =๎<?<+ฮ<Fภ<ญํฒ<๐ฆ<฿๒<-1<.<ะปy<weh<ฃ?โsฟuฤด>QJg?แำ<kตฟไ ฟดmฌพC&ศ>ฝฬ?ฎ#๛=&๓}ฟ?,"?0i=บ๑พ?G$?๒?ฟIำ>ฝ1<๋ฟ%X~?ๆ(ฟ๋สศพา๋~?ลTตฝว7}ฟฌุพ=hp?ิ?ข>FHฟ)ึdฟp CฝiNN?iิu?ีื>?ณพูหcฟ71zฟฒc(ฟำธ๚ฝE๐ี>ฒณO?IR}?้st?ฝ A?<มๆ>์?=6fพ28ฟ	<ฟๅเcฟcตyฟย๛ฟUyฟHhฟD{PฟJh3ฟfฟ?,ไพ๏oกพ$ื@พ\ฝT>=?]>ต!~>Z์ซ>ธ`ิ>ค๘>~?ม??ุ'?บศ2?๎ฆ<?เPE?x่L??S?cZY?ฃj^?ิb?ฌf?ํj?ฃํl?ๆto?0งq?rs?D7u?งv?ๆw?ฦ๚x?ต๊y?ธบz?o{?P|?พ|?}?ศm}?แล}?3~?PT~?~?2ฟ~?*๊~?a?/?K?ถc?งx?ส??จ?แณ?พ?๊ฦ?ฮ?1ี?ํฺ?ๅ฿?3ไ?ํ็?'๋?๒ํ?^๐?v๒?G๔?ู๕?5๗?c๘?h๙?t^?เrrฟ`?ผต>`ม6ฟ?ฐ}ฟศ้ฟถWwฟึ?ฟ)!๎>]9o?q@ฟฑไพu?o?๔}ฟฐ?o?S2pฟษ}?Kuฟ{ ?อ๗๛=๗3aฟR?ึ๏>๛~ฟศผl?3๙T>ธeฟ๗7ฟุAฏ>#ํ|?ํh+?'AพI๐]ฟ.vฟฟบ>x}5?วท{?9n?ิO ?|ไD>ีปพ๋ๅฟฬา_ฟญU}ฟB{ฟู้_ฟ๐2ฟfฑ๕พ_ {พฮJผะR> Gฬ>ฦ?ใe3?กึN?kcc?ยq?มz?90?ุ?ap}?.x?๐์q?0ูi?าว`?๒W?ํL?แB?้M8?ภ.? "$?ฉs?Y?[+?	A?>โ๏>8฿>ผั>ขOร>Kaถ>ล9ช>3ั>&>ี>Xป>๏o>ศ_>4CP>F?A>ช4>T=(>ง>๑ฺ>tส>ำ?=Z๋=?=ด๋ห=ฮฝ=Nฉฐ=llค=ฏ=[l=ฉ=}ทv=9e=&ฎU=c?F=}9=Q8,=๚D =น%=ๆห
=ื)=ฉe๐<ถ฿<ห.ะ<]ปม<ขHด<ัฤง<W<หH<ั2<?{<%(j<ไ_?>y_คพผร๗พฃTo?9D3?"L	>ตMีผM>JKR?d?b?sKถพ)Vฟ0e?.ฒพ๋7พใฒ>ฑพ4h>?ถ>glIฟ~?ฟx๓พNฟUาu?|aถ={๕ฟฉZxฝ็fz?p๕แ>7?2ฟ	pฟ*4พฯ$>?g{?+.?>rพgZฟ)]}ฟ4ฟ`:พฮtป>G?"9{??ฅw?ๅ๋G?๘>*a>-Dพrh๘พ๏b7ฟ]`ฟณ0xฟ๛ฟำzฟ?ฝjฟCHSฟU?6ฟ
ุฟไ4๋พbจพู;Nพ?๓ขฝ็=X{>3t>mง>่Uะ>9๕>มโ
??n?4O&?qฌ1?ญ;?ฐuD?(L?ๅR?แวX??๊]?8eb?ซKf?ฐi?Uคl?35o??oq?j_s?u??v?ณฦw?฿x?ำy?Eฆz?O]{?๓๛{?o|??|?สc}?9ฝ}?ณ
~?ัM~?๕~?Rบ~?๑ๅ~?น?r,?สH?Va?v??๔?รฆ?ธฒ?ฝ?ฦ?ะอ?ิ?]ฺ?h฿?วใ?็?ึ๊?ฌํ?!๐?A๒?๔?ฑ๕?๗?E๘?N๙?๘หb?IพRฟ+%K>k?ี8ฝT,ฟีฺOฟ&ฟฏ'6ฝฉbX?sร(??[ฟ๊C)ฝFฏ;??พxฟk๖?2ำฟ.|??Xฟน?ฬ>Mด>dcvฟ#ำ/?ท้>?ํvฟJภ:พีz?j๊ฐ>๏จTฟxฤKฟRm>ฑ9w?F=?lพลฝjRฟด{ฟ^ฟ=3d)?Eqx?ปs?:*?%u>ะSVพ๎SฟยฦZฟา{ฟว|ฟชcฟu๐7ฟ;๓ ฟพ ฝkํ;>0ย>z}?H0?bPL?๋a?Kp?dz?|๊~?๎??ฯ}?H6y?ฏฑr??ฝj?ฑยa?X?ำ?M?NฒC?้_9?9)/?(%?ดq??	?3 ?รจ๐>+แ>ิา>๖ณฤ>ฐท>ฃtซ>๘>๎3>๑>ถญ>ชดq>
6a>bฮQ>mC>6>.~)>ฎา>ภ๑>Dฮ>ชท?=ํ=(ธ?=ๅrอ=c:ฟ=}?ฑ=7จฅ=ฏ-=~=n=สx=ดTg=IW=ุYH=ss:=-=Ny!=ญD=๔ึ=_"=>4๒<dแ<lฟั<00ร<ฃต<ฑฉ<ะK<h`<7<f}<า๊k<๚zํพคV?n้zฟt^ศ>=ฝ?UO=?ฑp??B?*ฟ?๛ห?@ฟณ$ฟศ?Y.ฟบr>y<ฐoฝYฮ%พkd	?yฟjฟาo?ส๚พง:ฟฦลc?ห>้ด{ฟ}!ZพI;p?๕?๗ฟฎyฟ~ๆพู,?ฬอ~?ขN?EทHพษOฟaMฟณ๑?ฟ	๚vพa?>o๎>?x?kTz?|N?}๐?G8>๋!พ้ ๊พ/2ฟท&]ฟมvฟlีฟๆฆ{ฟสlฟVฟษ9ฟ๒>ฟ.๒พhKฏพ+[พ๓Eผฝณฟ<l>->j>๋ข>Gฬ>ฦe๑>๋D	???ฤ
%?0?คฑ:?C?agK?)<R?c4X?6j]?๕a?๊e?ฒ[i?Zl?๕n?#8q?/s?ใt?n^v?งw?)ฤx?[ปy?ซz?sK{?y์{?x|?๒๐|?ธY}?ด}?$~?EG~?I~?hต~?ฐแ~?
?A)?F?๑^?t?6?f?jฅ?ฑ?ผ?,ล?อ?โำ?ฬู?๊??Zใ?1็?๊?eํ?ใ๏?๒?๋๓?๕?๑๖?'๘?5๙?ค๚ด=7dฝM8ฟjk?<ู%?u&ฝ0]พำMฯฝ๖r?>bส??Dl>/ฟ!พ>?ญี>hปRฟฯศr?Wvฟ[ฝe?ศอ*ฟด>~?ฑฟU0?๚??^gฟงฏพฮฬn?ฦท๓>งา?ฟถa]ฟ#๎๑=ฌn?ัฤM?/oปกDฟ5B~ฟ๖ฎ$ฟ%4;!?Bt?*=w?I4?`ึ>Nฟ(พ}ฟ|dUฟ๘yฟ<~ฟ๘@gฟ5=ฟ๕ฟ?พฃtฝซ%>?ิธ>`?kส,?ะปI?บจ_?r?o?LJy?ณ~?h๛?(~?ษy?qs?
k?ฮนb?๛Y?O?รD?-p:?F50?.&?อn?!?)?	?a]?าN๒>Dปโ>c?ำ>ฺฦ>]?ธ>8ฏฌ>ธก>H>ๆ >๕>ys>$?b>pYS>ฦ?D>Z7>๖พ*>ม?>>า	>?M >xเ๎=+\?=๚ฮ=คฆภ=งOณ=?ใฆ=ซS=น=2=lz=,i=๙ใX=LุI=hื;=ๆฮ.=กญ"=?c=โ=็=ำ๔<	ใ<Pำ<ฅฤ<?ถ<Jช<Hx<x<;;<ฑh<ญm<?~ฟeิ?@Iฟ&ษพsC??ษ?ถt?eฏ~?o^?ยง%=ชyฟo6ญฝ5ดm?๒ฃhฟัZ?aขพณ>ำแแพ8ฐ>? ้|ฟ+กS?:๚]ฝ'Zฟ^nI?P?>ฯpฟ\ธพ๋"a?[)?ู ฟ 5~ฟd*นพธL???|ํ#?x๎ฝ?Cฟu?ฟ$JฟqEพคฬ>
ภ5??u?|?:ฎT?k?๎\>s๑?ฝr?พ๗,ฟYฟีซtฟฟ?|ฟวผnฟฟฆXฟอใ<ฟฟl๙พ+ถพ}่hพีฝ?A<S๕=JC`>๓d>^4ศ>Oภํ>ฅ?ฅ?ฤ#?ๆn/?บด9?.ปB?๛คJ?&Q?่W?ฐ่\?>a?ศe?/i?&l?nดn? q?M?r?Jนt?ป9v?Cw?จx?jฃy?๋|z?v9{?โ?{?j|?=ๅ|?O}?ถซ}?๛}?ญ@~?|~?uฐ~?f?~?S?
&?>C?\?or?g?ี?ค?aฐ?ป?Jฤ?Kฬ?9ำ?9ู?l??์โ?ัๆ?1๊?ํ?ฆ๏?ื๑?ฝ๓?a๕?ฮ๖?	๘?๙?ีYJฟG?>ฯฟ๑ด>pF~?c?C1ฐ>ฤ๖>ฮa?ลปb?๎ธพ๐ะnฟu8?5?=ฦํฟถึI?๐Sฟ';?พแพ๏หฝ?บ=?4v|ฟ๖ำญ>๕_A?๕Oฟ๗ไ๛พ๔า]?ร?j'ฟVkฟ@่?;]c?๒[?ถ?ถ=_ท5ฟAีฟ็2ฟ\~ฝ?/o?z?=?ช>4ค๕ฝUeฟฎOฟ?ษwฟ฿๑~ฟ)jฟ
Bฟแ?ฟฃพpยชฝ!U>Y?ฎ>฿5?g)?G?๔ถ]?๋m?ฺ}x?เ@~????ทx~?VUz?+t?ฑyl?(ญc?ัZ?พP?อาE?ฒ~;?็?1?Z2'?๓j??๘?ๆ
?;?๔๓>IJไ>gxี>M{ว>?Mบ>้ญ>ขFข>๎\>ต#>>B>u>d>^ไT>_NF>๖ฑ8>ฎ?+>ฦ( ><>ษี
>เ?>]ฃ๐=& เ=1ะ=฿ย=อขด=ยจ=คy=eก=๓=ZF|=ขวj=฿~Z=ฝVK=[;==/0=๔แ#==ํ=n=hั๕<มไ<ชเิ<ืฦ<Yธ<rซ<ฟค<ฃ<p?<~ฆ<,po<๖ฯฟqP ?e๘ฝ{oฟVํ=
ฎJ?K]p?ไM`?;;๑>ำธํพวwฟ๒kธ>๚1?tะฟuS?xฟL?ธ.ฟ$ฦe?ฝ~ฟ)?+?yฉ)>`หpฟม'?4L?เ^ฟ?พmM?!A?๋Sศพธ?ฟmK๋พ??:๚~?ืQ4?ไ๊ฝ7ฟrฟDTฟ~ถพ]Q>C,?ฒaq?ใ&~?Z?ืฎ?ัค>[มนฝฤแฬพ~์&ฟยยUฟ<ฌrฟอฟัz}ฟฏpฟป7[ฟบ๎?ฟ์ ฟ7๕?พฝพ6/vพูา๎ฝจ9v฿=ฉBV>S?>หฤ>ไ๊>X?ิ?|"?ฉM.?Oถ8?เ?A?ZแI?
็P?s
W?Kf\?a?ำ$e?ฐh?Eลk?[sn?|วp?8อr?ฐt?ฤv?1gw?ฝx?My?hz?W'{?-อ{?ๆ\|?qู|?\E}??ข}??๓}?:~?าv~?yซ~?ู~? ?อ"?o@?Z?Tp??A?ฑข?2ฏ?บ?gร?ห?า?ฅุ?์??}โ?rๆ??้?ึ์?g๏?ก๑?๓?9๕?ซ๖?๋๗? ๙?กHqฟxหw?ึB-ฟ:?พ่IN?z?	W?๚d?lW?ฤ??ำ,ฟsW.ฟ้ผp?Xคฑพpพ๙ฮ	?Cฦฟๆb ?๊:พlฑพฎห`?/mฟ๒>ฌท]?จD2ฟท๐ ฟ"G?4?j\ฟg-vฟๆฝoU?ฆ์g?zฯ9>p%ฟ:ฟฟห@ฟ1T
พ?=i?!}?KIF?:ึม>ฏHฝว!๘พูฅIฟ(Fuฟ&ฟLปmฟู๋Fฟถฉฟี[ฏพ?ฝู๑=ฅ>ฌ??>๗%?phD?บท[?l?ฆw??}?ู๛?หม~?A?z?฿t?อPm?ธd?[?า Q?ฺ฿F?w<?I2?w5(?$f?ั๊?sฮ?4?t๕>ขุๅ>?๐ึ>O?ศ>ป>#ฏ>Pmฃ>&q>\&>>ยw>?*f>*oV>?พG>ร	:>T@->พS!>๊5>}ู>ฺ1>5f๒=คแ=Kา=ร=ํ๕ต=[ฉ==ณ=ณ= ~=l=ร\=-ีL=M>=ve1=E%=ก=๘=๕=๛๗<pๆ<Iqึ<ฉว<sดน<Rะฌ<7ั?<@ง<ฅC<ฃ<ู2q<บซ>พ_u<?ผgฟBฟ?@[><+
?๓ไ>๙ฬฝ(CXฟh?<ฟu;?
$ฎ>๙pฟส?x?พWฟXaJ??z]ฟผ{?pฟ๙๔>Kม>=u}ฟน๒?>ณบ7?]Gฟ/M ฟ5?2V?lwพZa~ฟลZฟฤุ>ภ{?>\C?16=1)ฟHง}ฟิ]ฟ@ำพฯํ>โ๋!??๑l?.H?ํ_?$ท?ฑฆ>ถดhฝฃ๑ฝพ$!ฟ?าQฟMpฟ฿~ฟo5~ฟTrฟ9ด]ฟ๊Bฟ?1$ฟผ`ฟฮรพ]ตพืพ?e8ผษ=<L>JN>tภ>iๆ>ง_?(ง?ำ2!?ห*-?eถ7?7๛@?I?ี:P?tV?ใ[?/?`?/มd?^Yh?฿yk?ำ1n?p?ษr?ศct?๏u?ๅFw?บpx?sy?๔Rz?{?[ฝ{?0O|?อ|?;}?๐}?"์}?W3~?q~?tฆ~?ปิ~?ั?~??=?งW?6n?พ?ช?Pก?ฎ??ธ?ย?ภส?ใั?ุ?k??โ?ๆ?้?์?)๏?k๑?_๓?๕?๖?ฬ๗?ๆ๘?iพ๎ภ>อ]๚=๚rฟ>>ิb?.2?ix}?ฯ<N?U)=qฟYพwI?}ใ-ฟE>Wh>ธ ฌพเj>ฑEบ=jpฟ|Gw?์&Rฟฝ๙?q?่๔ฟF?ฟ?ฅ,?L?ฤL?พ;๚|ฟ{gพ;E?
q?=J>n์ฟF ~ฟALฟำผPพฤัไ>กpb?ฑเ~??{N?ภฌุ>fr๒ผ๒ๅพNCฟ๘nrฟเ๎ฟpฟKฟdZฟปพฉพฟ๊ฤ=๕>ซy๗>y"? ชA?(ซY?บk?รv?-o}?๖๎?X?มZ{?ฒu?Z#n?{e?\?J'R?ม๊G?y=?ืP3?g7)?_`????คต?ื๔?<๗>Lf็>ษhุ>฿@ส>ฬ้ผ>=]ฐ>มค>->?(>๔u>วx>{ัg>ิ๙W>@/I>za;>่.>ง~">L>)?>อ#>)๔=?Gใ=^ำ=B๋ฤ=Iท==ช=ล=ดฤ=p=?๚=:n=ฅด]=SN==@=ฝฐ2=J&=uภ=%={=n๙<~่<็ุ<{ษ<fป<2ฎ<ฎ?ก<?พ<ูG<ศ<๕r<Ay?4!mฟq~?X6กพ{ฟ๋[ํพ*/ขฝ0พชฟ๙วฟ5ฌพ์?t?ฟฝ.แ;ฟFใ}?ึVyฟ~q?02yฟ]๐~?฿๚Qฟซ>,1?xฟฦง>ฝ]T?>ฺ)ฟ=ฟ??ฺf?วใพ_yฟZn#ฟ`ฉ>,Xv?ุ๏P?cn?=Rฟ๓zฟeฟฃึ๎พ๚ฟ=T?6๒g?Kใ?}๕d?G%?vค>ึCปผใะฎพิ7ฟฒบMฟb4nฟMฯ}ฟ[ะ~ฟd๙sฟ?`ฟนีEฟl'ฟใพฟๅสพ7Mพพาศบผ๗จณ=51B>๒ฝ>nๅป>rธโ>บ?ฅ0?X็?N,??ด6?2@?lVH?O??U?็^[?-`?฿\d?h?๓-k?ี๏m?;Up?jr?8t?	สu?^&w?Tx?Zy?พ=z?ด{?lญ{?aA|?ม|?ฒ0}?๕}?Zไ}?,~?.k~?eก~?Zะ~?๙~?@?ร:?0U?l?ๅ??ํ?อฌ?๓ท?ม?๘ษ?6ั?{ื?้??แ?ฐๅ?7้?D์?้๎?4๑?0๓?็๔?d๖?ฎ๗?ห๘?O22?
ฟะU?jbฟฑฟฏร>p?>?	/<?};ป>pPํพfW}ฟE >a?Dhฟฮฃ๒>?Gใฝฅ0ฺผฆZฝeธถ>UคBฟ ์?p-ฟfพ๏r}?ัEฮพ(ๆXฟี?ฤz`?ม๒พใฟาoฌพํ`2?|ีx?-xธ>vLฟKzฟe>Wฟฌพcฦ>ะZ?า?๗V?ร?๎>aฝ<?ัพfฉ<ฟEoฟ๓๛ฟFBsฟ%Pฟาํฟฃวพฒพ_ใ=>?๎>?ํ???>?`W?ฃi?อิu?U๗|?Wู?Z=?ัำ{?a8v?V๑n?mpf?m]?#+S?๓H?ต>?#W4?'8*?ฃY ?_ฬ?1?๕ะ?ป?๘>G๓่>&เู>?ขห>87พ>ฉฑ>๖นฅ>>5+>ดg>(z>๐wi>^Y>J>น<>kม/>ฉ#>"c>หเ>น>ร๋๕=?๋ไ=hี=jWฦ=ธ=๕าซ=|๋=Xึ=+=๊=๒๓o=O_=าO=,gA=?3=ๆ~'=e฿=/=?==๛<๙ฬ้<ู<Mxส<Yjผ<Vฏ<%*ฃ<yึ<L<ํ|<2ธt<ฬ7?G[ฟ๛ว?ธ๏๎>4Xฟ?ฏlฟMv*ฟ8-ฟฯDnฟืbฟณ1>Sุ|??า๓พซ2ึพdma?/k~ฟภ่?&}ฟ$o?ำH&ฟ.ผษ<~ฅ<?ฟvฟ3>Nj?๛ฟศCUฟ&!๖>s?x๓ผ\qฟA7ฟฏp>ฮn?๒\?#Q>
ฟM`vฟ๔kฟhเฟN=R?วeb?่๗?Bi?-
-?2ถ>426<cพ:)ฟ	{Iฟ?ผkฟ2๒|ฟKฟิuฟฯnbฟQฑHฟข*ฟ?
ฟHัพ??พ.พ*ฌฝบ=ๅ 8>e*>ฬรท>฿>?Oธ?#?6เ*?ฒ5?ึ5??!G?"฿N?5DU?ู่Z?น_?แ๗c?-ชg?แj?cญm?p?เ7r?t?Fคu?w?8x?้Ay?b(z?/๐z?_{?y3|?ต|?A&}?่}??}?ฯ%~?Me~?L~?๐ห~?3๕~?๑?ๅ7?ดR?์i?~?s??ซ?็ถ?ณภ?/ษ?ะ?ไึ?g??,แ?Nๅ?โ่?๛๋?ช๎??๐? ๓?พ๔?@๖?๗?ฐ๘??๐z?/า~ฟฤu?6พ&wฟณฎพ(*D>๙`>WY?พ#XฟฌNฟธ?ั|?อฟตาA?๖,฿พุ4>'ณพภ??๖Efฟม=z?"?พ0*๊พฝฎ?ฒ@pพ๗UlฟZsึ>๐ภo?7พg?~ฟ฿ีโพหง?r}?vไ>?h?พํuฟaฟคญพฌง>๎cR?l๔?฿)]??`?พิy=kคฝพRบ5ฟhษkฟZฝฟฒจuฟZTฟ๐b#ฟัำพๅค5พU=๊>+ๆ>ืU?<?jU??h?X?t?u|?๛บ?ัo?oF|??v?ปบo?Tg?
^?Z,T?๚I?(ง??๚[5?ต7+?ํQ!?๗ป??ฌ?๚>๊>๕V?>ฆอ>Gฟ>ษฯฒ>๎฿ฆ>จฌ>f->TY>O|>;k>ล[>ดL>ฆ>>?1>Oิ$>ญy>dไ>>xฎ๗=ฎๆ=kึ=รว=0๏น=จญ=hก=๙็=ไ=ฉื=]ญq=b๊`=rPQ=หB=FG5=4ณ(=T?=9=๕=ฐ?<t{๋<"#?<ํห<Lลฝ<๐ฐ<Vค<๎<BP<o<?zv<คJพGSฤฝาNพv?ทพ	tฟ>B{ฟสพyฟข}{ฟ/	ฟQ?ศQ?โ^Kฟ. ฝ๛<'?fffฟ&Wu?ัoฟM?นฒ฿พ๙Wพ๓]?2จcฟส้Kฝญฺx?ภษฤพ[vhฟ}ณ>-ฺ{?)๛ภ=6eฟฺฑIฟื>๖3e?rMg?ื>H$๔พคํpฟๆqฟQูฟญ๙ผัํ ?๐O\?๛?ตสm??N4?ผgว>ญ8=พ๙ฟฒEฟ*iฟฎ๐{ฟฺฆฟร๓vฟvฌdฟฉ|Kฟ1ฝ-ฟrbฟ,๕ืพjพ?น)พ๏;ฝฦ=?.>ผ>ฃณ>์J?>xา?>+>?8K?ธ)?รญ4?"Q>??ฦF?ง/N?ูชT?TZ?YD_?8c?ฑQg?j?|jm?aแo?fr?<แs??~u?ไv?}x?)y??z??z?5{?w%|?hฉ|?ผ}?ฬ~}?ิ}?๘~?`_~?+~?~ว~?Z๑~??5?4P?ยg?(|?ำ??_ช?ูต?ษฟ?dศ?ุฯ?Lึ?ใ??บเ?๋ไ?่?ฑ๋?j๎?ล๐?ะ๒?๔?๖?p๗?๘?๑น>็+ฟุNั> ?"ซaฟ฿Xฟ:ภแพ?<ฟพฤ.ฟลฟ่฿พธม_?ษj>i2pฟ๕รp?ห 6ฟwฦ?ซHฟJ?){ฟวf?คศพa!ฟx?ธqฝมPyฟพ>฿z?@ฝS๊yฟ4ฟ?ฝา?ฦิ??ฒพ๘nฟ?iฟ์Bฮพำ>ถ2I?G?ลc?ณ๒?พcู=ฆlฉพ.ฟO?gฟ*3ฟ<ะwฟ xXฟฐธ(ฟte฿พ}Mพฟ๖<~y>c?>?ฑ?9?ฏ6S?ษf?พึs?ฟ้{?ๅ?ป?ฒ|?[{w?p?ะ4h??^?์*U?y?J?าฌ@?X_6?6,?<I"?ยช?\g??!?>,์>3อ?>?eฮ>๘ะภ>ด>จจ>ภ>o/>ิJ>พ~>\ฤl>
\>ฦM>h?><B2>?%>,>๓็>z๙> q๙=x3่=f$ุ=ง/ษ=<Bป=WJฎ=Q7ข=๙==ฤฤ=ฤfs==b=?ฮR=/D=6=็)=C=C$=๎=@ฺ?<๎)ํ<พณ?<๏aอ<> ฟ<ฯ?ฑ<ฅ<ฒ<vT<7a<=x<[nฟ^ฒ=?กiฟgW]?ะฝ๑>sK	ฟลeฟ์xmฟB;ฟs-ฝไ8f?ฟ๘>m2yฟฑ>?ญ>ป4ฟ?pR?ถKฟ?+ยHพ?พ?}t?ฬ?FฟNMtพ๖?พyhพ(vฟผWZ>ํท?F๎]>s่VฟยpYฟd3=Y?์o?pท>ฝัพภNjฟไำvฟ๋Iฟ๛ฝ[^๊>`ดU?ภ~?์q?}L;?4}ุ>?หก=ฺqพ3ชฟ@ฟถXfฟๅสzฟRโฟ
Ixฟผิfฟ7Nฟำ0ฟtงฟv?พX๎พ|>6พจ+kฝc=b๒#>๚>vฏ>ฉื> |๛>>ย?๚?>(?๓ง3?k=?๊?E?M?T?UอY?เฮ^?โ+c?๘f?Gj?!'m?ฺฆo?าq?ตs?๔Wu?dรv?ฌ?w?y?4?y?มสz?๎|{?[|?.|?%}?u}?ซฬ}?~?iY~??~?ร~?yํ~???2?ฏM?e?Dz?0?ต?%ฉ?ษด??พ?ว?'ฯ?ณี?^??Gเ?ไ?6่?f๋?)๎?๐?๒?j๔?๘๕?P๗?z๘?ผyฟภE>ปO?พCภx?Xฅพำว}ฟืdฟฦUฟ|wฟb๒bฟตซ=ๆ?ย	Bพ<ฟู๔?<gฟจ?K?้ฃOฟชRm?ฒฟI๖E?Hคฝ9=Fฟ^{h?ฟศ๒=พ{ฟฑ?=5X?oำฐ=K qฟ๑"ฟ?>ญ~?"x??hพฮfฟ*ผpฟ,o๎พ๖O>vE??3ฬ}?&li??.?mต>ร็พ'ฟโcฟ]~ฟVธyฟhe\ฟ๎-ฟ๕๋พญ7eพ<e>gิ>` ?y*6?
๖P?u์d?
วr?T{?d?พ?H}?x?ณ?q?9i?fํ_?ึ&V?ญ L?ญฐA?<a7?63-??#?ภ?๗K?'b?รม?>ํ>แB?>ฦฯ>Iย>%Aต>%+ฉ>\ำ>O1>3<>:ึ>Rjn>-#^>ป๏N>yฟ@>3>ฝ)'>?ฆ>z๋>O๋>ฝ3๛=8ื้=Yซู=ผส=Cผ=ฏ=6]ฃ=3=O=?ฑ=) u= d=BMT=๐E=ห?7=ฯ+=0<=L/=ๆ	=hT =gุ๎<ZD?<ฟึฮ<0{ภ<ญณ<ฏฆ<N<ซX<[S<6 z<FOฟT.{?gฟห๔q>bAr? >Bๅพuฟ๙ดพ่์>Wู?๑ผ]=จ\{ฟญ-?ผๆ#ฺพIฌ?๙ผฟ=๚ฟ>!bk=SQ"ฟฯ,?จ๚!ฟก]ึพ฿1~?๐ฝ?~ฟ:u=D?๚ซ>ุrEฟ{ญfฟ๒ฝ)%L?Tฟv?๓~?>((ฎพแbฟ็ถzฟ&*ฟฆพั<า>N?ธ}?-๏t?H B?ศH้>^็=kaพ<ฟfื;ฟ๗lcฟyฟใ?ฟyฟm็hฟทแPฟ?3ฟฎใฟ,ๅพkขพ์ปBพ0ฝ;จ7=ติ>?บ|>Jซ>ัฮำ>)"๘>D?Mจ?cd'?ฎ?2?ป<?2E?pอL?:uS?มEY?ฅX^?เฤb?๑f?๙i?Qใl?๎ko?fq?ฏs?e1u?๑กv?จแw?๒๖x?c็y?ืทz?l{?&	|?฿|?z}?`l}?ฉฤ}?%~?fS~?ห~?พ~?้~???+/?&K?ac?]x??G?้ง?ทณ?๐ฝ?หฦ?uฮ?ี?ฺู?ิ฿?$ไ?เ็?๋?่ํ?U๐?o๒?@๔?ำ๕?0๗?_๘?ฟz๘f?#ewฟๆW?Wั>มื,ฟฆฆ{ฟ๒ฟ?tฟ@a	ฟ็??>;ไk?%ฟธึพdm?ถ~ฟ๚q?(%rฟ?p~?lsฟFJ?D>cฟlไO?Eซ>ศซ~ฟs๔ผ|W?D_>้?dฟd8ฟ-Cซ>|?Lล,??m:พ$]ฟvฟ'นฟM~>uฅ4?a{?ํn?!?ฉgH>พฤJฟ๒x_ฟธ<}ฟ`{ฟ๖!`ฟ3ฟb๖พHั|พษ๙eผ9P>?ห>C?'*3?ตจN?Bc?Iฌq?pดz?+?ใู?~w}?cจx??๛q?ย้i?๕ู`? W?ฎ M?บฒB?ฅa8?&/.?ๅ4$?๏?๊/?(<?a?>F ๏>?ท฿>ๆ&ั><iร>`yถ>cPช>kๆ>3>q->ภฬ>p>.ญ_>_P>ยB>ฤย4>^T(>ฝ>๖๎>?>M๖?=๎z๋=D2?=หฬ=D่ฝ=ฉมฐ=ค=ห==๔=ูv=์บe=งหU=ู๖F=)9=P,=[ =T:=฿
=ฏ;=เ๐<๖ิ฿<Kะ<"ึม<aด< ?ง<๊4<?\<E<แย{<ลk=ฤอ?>กพH	ฟๅฌi?ื<?ื๒;>?จ<[>X?p?]?ฏ่ฦพx|Qฟfh?ฦกฟพ?อฝญ!ง><'ฆพ??แ= S>ิGLฟg{}?	N์พZcฟaัt?5iะ=็โฟขลฝDืy?๒Xๆ>๏[1ฟ์@qฟT&พ|่<?บธ{?น๗ ?พดฎYฟY}ฟ@d5ฟTเ>พ๎น>r?F?ฎ{?๑ฺw?gH?บฤ๙>9	>ฌนAพd๗พ67ฟj[`ฟ3xฟ๙ฟฃzฟZไjฟ?zSฟ๏ฺ6ฟํฟต๋พpแจพฎ1Oพลคฝ?ฌ=ณ>B|s>สง>sะ>ฤ๔>ล
?UT?๘7&?๗1?;?ๅeD?ทL?๙ุR?RฝX?งแ]?2]b?ฏDf?ชi?l?0o?เkq?๓[s?
u?Bv?pฤw??x?kัy?ฬคz?\{?ุ๚{?y|?ผ๛|?c}?ผ}?(
~?YM~?~?๘น~?ฃๅ~?u?8,?H?*a?rv?แ?ื?ชฆ?ฃฒ?ฝ??ล?ยอ?~ิ?Sฺ?_฿?ฟใ?็?ะ๊?งํ?๐?>๒?๔?ฏ๕?๗?C๘?/g๛พฅb?๖kRฟแI>ฦk?ฝ1ฝ<,ฟหOฟภ%ฟ>ฑ0ฝX?t(?ช?[ฟ+%ฝป;?,ฑxฟM๗??ิฟภ|?๋Wฟ๗ห>V?ด>ณlvฟ้ป/?้๋้>ๆvฟ%;พนz?Aฑ>aTฟ'ะKฟm>Y5w?ป=?$Kลฝท
Rฟ){ฟึฟใร=?\)?ox?y s?L*?ฐu><8VพฃNฟรZฟั{ฟZศ|ฟิฌcฟฅ๓7ฟ้๖ ฟ?#พ?ูฝ์฿;>Gย>?z?F0?ีNL?ษa?p?๒
z?O๊~?๎?7ะ}?ค6y?%ฒr?hพj?Hรa?ฅX?x?M?๔ฒC?`9??)/?<)%?Nr?4?	?ป ?ฤฉ๐>,แ>บา>ฮดฤ>Mฑท>buซ>G๙>4>>Iฎ>พตq>7a>SฯQ>๓mC>ํ6>๑~)>cำ>i๒>โฮ>ะธ?=ํ='น?=ำsอ=A;ฟ=K?ฑ=๗จฅ=a.=ฒ~=	=๊x=ภUg=
JW=มZH=Kt:=e-=	z!=\E=ื=๖"=X5๒<eแ<_ภั<1ร<jคต<vฉ<L<a<ค7<}<_?v๎พปอ?{ฟwว>gย?Z=?ญย?wC??ย?#?ฑ@ฟ"่ฟฒส?F.ฟ์r>๐{<]ฝ&พ[	?Gฯjฟฬ~o?ณธพ':ฟ3ธc?)K>5ฐ{ฟZพ3p?1?:่ฟืyฟศพT	,?1ฯ~?ฐY?HพEยOฟ.Nฟb๘?ฟwพQ?>้>?ฏx?ไUz?๎N?ต๕?i]8>6ึ!พ)๊พๅ	2ฟ$]ฟฏvฟJีฟง{ฟQหlฟ#Vฟ}ห9ฟ Aฟบ2๒พOฏพF[พTUผฝ:[ฟ<ห>8j>Q่ข>กDฬ>c๑>๏C	?ต???	%?ะ0?ฑ:?C?๋fK?ย;R?	4X?่i]?ู๔a?ี้e?~[i?SZl?ไ๔n?8q?๋.s?}ใt?X^v?งw?ฤx?Lปy?z?hK{?p์{??w|?๋๐|?ฒY}?zด}? ~?AG~?F~?eต~?ญแ~???)?F?๐^?t?5?e?iฅ?ฑ?ผ?+ล?อ?โำ?หู?๊??Yใ?0็?๊?eํ?ใ๏?๒?๋๓?๕?๐๖?'๘?K|๏>3ก>ฦ,สฝFฟ)p?Nึ?๎ฺฒฝฉซพๅีพ์>E?V>คN~ฟฤฐ>,ุแ>M:Vฟ=t?.wฟb์g?ผF.ฟ??/>dฉ?๚cฟ-	?๋e?kbhฟ#ญฉพgฝo?Ja๏>มTAฟ่M\ฟรี >HUo?3พL?J "ผXEฟ~ฟ!ฒ#ฟฬC<u?t??v?ฅ3?*<>ฤะ+พ?ฟรUฟ๑zฟz๏}ฟ:gฟ๒ม<ฟฑฟ๗ษพ#nฝW*'>|น>ง?-?่I?ษ_?ำUo?Wy?]?~?ษ๚?p"~?^ฟy?bds?&k?[ฉb?
Y?	๚N?ZฑD?๙]:?X#0?&??]?ำ๕?๎	?N?2๒>}?โ>ๆำ>  ฦ>์่ธ>"ฌ>๐ก>๛5>>ท>3[s>ฦภb>๔>S>ลD>C7>tฉ*>ด้>ำ๕>?ภ	>ค= ><ย๎=@?=ิ฿ฮ=8ภ=้8ณ=าฮฆ=๕?=`}=y=FLz=๐h=kศX=งพI=ฟ;=ฏธ.=๕"=bP=ะ==
=ฯใ๓<+๖โ<.5ำ<ฤ<G็ถ<๋4ช<"d<Fe<ศ)<8H<฿Db?฿่|ฟโฟ~?[RฟmฏพXK??๏4q?&m}?c?๋=wฟ!๋ฝลKp?บฟeฟั&??Eพ$จ>ฐลุพ";?;2|ฟงไU?<<ฝ?$XฟbpK?บษึ>qฟ^ณพLb?ฮ'?Idฟ;๗}ฟขธตพญ?ฬ??}ศ"?๊t๙ฝ๐าDฟ?ฟูIฟ]Jพนฉ>ณa6?|u?ๅ^|??FT?ช??{Z>Sษพb?พg๏,ฟ๖ศYฟฏฬtฟ%ฟุ|ฟ%nฟ๗yXฟฏ<ฟถaฟ$ฃ๘พวตตพ5hพต?ำฝคญN<(ส๖=ษ๎`>ทฒ>hzศ>?ํ>ม?qง?{ฺ#?</?ปล9?สB?ฒJ?Q?ๅฉW?f๑\?ีa?ee?๏i?&l?ฦธn?ษq?s?#ผt?3<v?gw?hชx?ฅy?P~z?ฌ:{?๎?{?kk|?ๆ|?BP}?Mฌ}?
?}?A~?๕|~?หฐ~?ฐ?~??B&?nC?ฑ\?r??๐?&ค?uฐ?ป?Yฤ?Xฬ?Dำ?Cู?t??๓โ?ุๆ?7๊?"ํ?ช๏?ฺ๑?ภ๓?d๕?ะ๖?๘??."3ฟฌ1?]?|ฟk่>?้y?eg?=๓>ว๘ะ>_ไW?,ฃj?ญ?พtฟw-?qป=Cฟ'์P?๏Yฟx๘A?ม๒พrฝฎ8?เ}ฟaAป>.๏<?ๅSฟ|ร๑พ์u`?ท?๔+ฟk?iฟW<	e?#Z?์๑=ฑี7ฟCทฟp1ฟfฝ@๛?.๎o?จ+z?ฦO<?ตSง>พฒฆฟzPฟ่xฟี~ฟk*jฟlAฟฟ [กพC'คฝถb>hUฐ> ศ?ฑ?)?vG?ฐ๛]?9n?lx?บM~?ใ??(n~?Bz?๔t?๛[l?*c?ญ๛Y?]๓O?๊ญE?แY;?1?ๅ'?H?วื?๎ฦ
?ำ?บ๓>?ไ>๚Dี>ะJว>< บ>ฃพญ>eข>77>e >
q>| u>^Jd>yฎT>F>8>้ำ+>๘?>3๙>Vฒ
>ู>ำe๐=ิฦ฿=ฯKะ=)แม=tด=ช๔ง=Q=|=.f=?|=bj=สFZ="K=ฦ
==๘์/=฿ท#=i[=ศ=๑=E๕<ลไ<?ฉิ<๔ๆล<%*ธ<aaซ<ฝ{<yi<ํ<q<ํ๒ซ=พโ6ฟวM8?๊?พdฟน๖]>ขX?๚7w?ูดi?๓	?คมฬพ:{ฟ>ฯD<?๏์~ฟ K?ค๑ฟ"P?-'ฟ`ea?wฟื๊1?ึ,>ฆCnฟ๚ฝ,??4?ฉaฟา2๖พษ_P?,r>?fGะพ๔ฟซไพ]??H??า(2?eํ@ฝW๎8ฟฟ3?RฟฒพHY>ฒk-?i๑q?๕}?บY??6X|>o6รฝต็ฮพถณ'ฟ'IVฟr๔rฟ%-ฟๅ^}ฟซVpฟD฿ZฟT?ฟ฿x ฟ|?พฏผพ?_tพ?`๋ฝม๕๔:irโ=x?W>z>?ฌฤ>๊>z<?N?nฉ"?;u.?ู8?o๚A? ?I?u?P?่W?$x\?'"a?^2e??ปh?ฯk?C|n?9ฯp?๐ำr?t?ำv?kw?x?y?฿jz?า){?Tฯ{?ร^|??|?ยF}?ค}?้๔}?๑:~?w~?(ฌ~?ฌู~???#?า@?oZ?p?ิ?x?แข?[ฏ?*บ?ร?กห?ฆา?บุ????โ?ๆ?๊้?เ์?p๏?จ๑?๓??๕?ฐ๖?๏๗?O๊?cญ}ฟ?ป๐Kฟq?พ8&c?ฃo? +C?< U?mภ?Zt?ฟณ?ฟญGh?ชMพรพ?])ฟ?ลrพ พZ?f3qฟฺฌ:>๗X?๎฿8ฟฟvL?/?/ฟvMtฟไตฝ?X?ศคe?c{&>ๅ(ฟๆฟ(|=ฟ฿x๗ฝ~๙?j?[ซ|?ณD?Nฝ>OUฌฝ็?พg๊Jฟัuฟzฟณmฟ๔๐Eฟ<zฟีฌพ(ัฝ๛=ัง>i? ?nญ&?_๗D?"\?ษำl?qำw?h๒}?k??]ณ~?4ภz?ึบt?แ$m?ณkd?๊Z?s๊P??จF?DT<?2?5 (?2?น?ฤ?ต๊?๕A๕>ชๅ>eฃึ>?ศ>=Wป>ใโฎ>ง0ฃ>I8>๑>BR>ฅv>าำe>แV>?rG>๚ย9>O?,>/!>?>ค> >a	๒=Mแ=รทั=4ร=ฐต=~ฉ=c=ตz=>S=๖พ}=/&l=(ล[=nL=V>=@!1=ษึ$=nf=ม=ศุ=ป@๗<^ๆ<หึ<ๅAว<mน<ึฌ<X?<ญm<<วf<?
Kฟ?	>๊า~ฝTฝ?vฟ๖ ์พด>ฎช%?[? _4=FHฟ5MฟQf*?N=ื>ฝ{vฟฅงs?aMฟภ??CLUฟิถx?kptฟP?Mซ>ฯต{ฟิ?|1?esLฟไฟ[ภ:?(R?  พ๒?~ฟ?ฟฒEโ>|?$a@?LIไ<K#,ฟษ~ฟ\[ฟ-;อพ}ค$>o$?็m??ื^?ถ?]๕>?จฝ*	มพิW"ฟธฅRฟ;๙pฟTฉ~ฟ~ฟน๚qฟฺ2]ฟ5NBฟI#ฟ6ฎฟiยพYพ|mพqผ?ฮ=_MN>u>>	?ภ>น+็>:ถ?๔??v!?ัf-?6๋7?)A?!EI?a^P?V?!?[?ฮท`?ยีd?Ekh?pk?Z?n?Pp??ฅr?คlt?8๗u?Mw?vx?xy?MWz??{??ภ{?R|?ะ|?1=}?ศ}?ปํ}?น4~?8r~?}ง~?กี~??~?6 ?1>?)X?ฆn? ???ก?@ฎ?4น?ฑย?้ส?า?0ุ???%โ?%ๆ?้?์?6๏?v๑?i๓?๕?๖?ำ๗?	:ญพ-ๅ๗พ<i?ษ์ฝฉH[ฟ๊_ฤ>>Qu?ฒป~?`๊?ะ(c?(อ<>rcฟ/"ัพฒว?5ฟIZๅ<ข>Whีพ๎5>r<[ฟnr?1คZฟ?ปBm?โ-ฟ฿7ฟ\4?`F?h๎พl}{ฟฐงGพใสI?ผ)o?i>}>฿?ฟฅ~ฟภIฟ5p=พส๘์>>ed?๗y~?WHL?oา>๙K,ฝU๊พ๛Eฟ}?sฟ??ฟkุoฟๅOJฟ/ฬฟ6ธพฑ?ฝLั=ึ>ิ๙>{p#?ลlB?๊<Z?k?ฒw?l}?b๓?๒~?J8{?_u?ึ้m?๒Ge?ึี[?H฿Q?{กG?"M=?^3?๐(??ช?v?$ธ?ศ๖>฿๘ๆ>Vุ>K฿ษ>๎ผ>ใฐ>ตBค>19>ดแ>^3>Jx>"]g>-W>ิษH>ู;>ฅ(.>[,">ี?>ช>0แ>ไฌ๓=aิโ=ฑ#ำ=?ฤ=ช๋ถ=O@ช=t=\y=L@=Ix=๚ภm=C]=O๊M=<ก?=U2=ฒ๕%=sq=น=ภ=0๏๘<๗ง็<ื<ิศ<฿ฏบ<Kบญ<๓ชก<เq<4 <H<็pฟH?_?
๑LฟH~?ษฟ๘klฟw\พ}ห=7lา<?์พo{ฟx๚๊พซi?๗ภ)=&MฟOๆ??ำrฟยณh?"ขsฟ๒๗?ฬบ[ฟ|ค>P)??ฟIภ>jM?wf2ฟ+5ฟะ!?mขb?บ]?พ={ฟพฟjถ>>x?ัYM?9Iา=ฝฟฦ{ฟ๎bฟ<E็พ๑฿=I?บ`i??ล?)c?b#?^>9๑ฝี?ฒพศ?ฟE฿NฟT?nฟล~ฟ?จ~ฟ(sฟt_ฟw	Eฟฦ&ฟOาฟตศพ}พ&พQพ?ผฒน=ณ๕D>๖?>ฝ>ฝใ>S.?๒?ษB ? W,??6?W@?H?ZฝO?bV?^[?หL`?xd?*h?็Bk?n?ep?ฟwr?Dt?cิu?X/w?N\x?May?Cz?ว{?าฑ{?1E|?็ฤ|?3}?p}?ๆ}?u.~?หl~?สข~?ั~?๚~?)?;??U?ชl?h??O??"ญ?=ธ??ม?/ส?eั?คื???ผแ?หๅ?N้?Y์?๛๎?C๑?=๓?๓๔?n๖?ถ๗?yฟ.ำ>SูbพW๒"?zฟ๑[พ"?ญ`?I\?ฤ	??พฎํฟถบr:ep?B๓WฟYธ>Lร;~
พKค=hั>M3ฟ}~?๐ไ:ฟtLJพz?ใ๊พีสPฟ๕บ?ศ Z?+*ดพอSฟฃพม9?เv?สฉ>*ึฟC๖{ฟึSฟW~พ!!ั>]?8??S? ]็>๚.ย8แJุพH?>ฟepฟ??ฟ๙_rฟ Nฟฟฃ|รพkพ
lง=ฃ>ุ๑>( ?^ึ??ELX?&j?6(v?ว!}?ศแ?5*?ฮช{?~?u?ึชn?ไ f?ะพ\?ูัR?xH?wD>?฿?3?ฤ฿)?ฯ ?y?วL? ?wN๘>~j่>อ^ู>๓(ห>Oฤฝ>ฃ*ฑ>Tฅ>๏9>(า>_>O๏y>Nๆh>\?X> J>คB<>ํR/>{B#>>I>Rย>]P๕=[ไ=ิ=?ูล=7'ธ=fซ=&=x=Y-=อ=ย[o=?ม^=/NO=u์@=ฬ3='=x|=ฒ=Rง=ฅ๚<8้<fู<ฤ๗ษ<ป๒ป<ภๆฎ<ยข<v<X๒<q)<Yeพ5i?ขฃyฟีqE?+(S>?tฟ?Nฟf๗พโฟฐฤWฟ9stฟSฐมผ๙??ใฐพ{	ฟ|เn?ื?ฟ?ฅ}?ะฟลฐv?ยธ6ฟYwู=|๒.?`๔zฟืกR>ฒc?ง ฟsMฟ% ??o?ษ]ฝ๚Htฟใ0ฟ`>คq?M?X?ล(5>ชฟC๚wฟำซiฟJ ฟกh=)?1_d?  ?	h?ทx*?2ฐ>ฌบาฮคพCฟo๖Jฟ
lฟB}ฟฆ$ฟำ?tฟคaฟ้ถGฟ')ฟ`๏ฟใ๘ฮพฮพ๑ูพTฟ๘ผyJฅ=ช;>ฉพ>ึ0น>Kเ>ษค?C:?6?ษE+?6?`??๚ำG?bO??xU??[?แ_?ฦd?ศg?๋๛j?Yฤm?s/p?3Ir?t?Rฑu?ํw?์Ax?lJy?ฤ/z?๖z?๋ข{?G8|?ถน|??)}?	}?9฿}?'(~?Ug~?~?vอ~?๖~??ใ8?S?ซj?ญ~???ฌ?Dท?ม?tษ?ฤะ?ื???Sแ?pๅ??่?์?ภ๎?๑?๓?ฬ๔?M๖?๗?$7ฟ0z?BๆbฟาG?]SฟjฒWฟMgผไ>Lณ่>ธ8=]5ฟJฟhฟฆ?ั>v<?ปฮzฟJi&??Rพฌผ!>อoพฝb ?I5YฟuT~?*KฟLฃยพศใ?&mพฦ
eฟฆ๓> j?.๋nพรฟ:กฬพDy&?ดํ{?8า>?๋พ\?wฟB]ฟ4พด>๙U?W??=QZ? ั๛>?-=n๋ลพeข8ฟDmฟuเฟอฑtฟvRฟ$!ฟงฮพนผ+พะ๒z=ษ>้ว้>rิ?Q4=?APV?ภh?Du?~ฌ|?ศ?ี[?พ|??v?฿go?๖f?	ฅ]?#ยS?I?A:??๐4?ฮ*?4๋ ?ำX?๔"?ฉQ?ำ๙>?้>ศปฺ>8rฬ>_๚พ> Nฒ>3fฆ>:>zย>D๕>็{>Woj>mkZ>9wK>]=>%}0>X$>O>฿x>nฃ>ห๓๖=ฬแๅ=x๛ี=ป,ว=ภbน=ๅฌ=ซ?=ฃv=c=tu=๖p=5@`=ฒP=ญ7B=พ4=3(={=ช==L?<'ษ๊<3}ฺ<ณRห<5ฝ<5ฐ<)ฺฃ<Fz<|ไ<ว
<3?ๅZ>qํพ?n=โ[Q?9แ	ฟ๕ฟR1eฟ7dฟฮฝฟu4ฟ6ี>้yi?A-ฟ2MพB?กฦsฟ_ษ|?ฏํxฟธz]?|ฟ้ฝงaQ?ัวlฟ
_๐<oยs?ยฒไพท'aฟLฯ>9๏x?ก๓/==ชjฟ่zBฟโๆ5>9mi?@8c?!> ฟ%TsฟCoฟฌฟ๑I;%ฑ??ๅ^?{ล?vl?ฅT1???ภ>๗"?<Fyพnฟ?๋Fฟฑ8jฟภ_|ฟใฟ^vฟpมcฟ[VJฟ>r,ฟ=ฟำ2ีพ๚ดพ$พ~\(ฝช?=x92>คz>Vต>๏ี?>ข ???&ึ?03*?ฦ5?ฐ>?าG?wxN?|๊T?Z?ศt_?hผc?jvg?{ดj?@m?๙o?Yr?l๓s?u?O๒v?]'x?e3y?ฬz?Gๅz?๋{?G+|?rฎ|? }?}?ๅื}?ฮ!~?ีa~?L~?Vษ~?๓๒~???56?>Q?จh?๏|??ต?แช?Iถ?*ภ?ธศ?!ะ?ึ???้เ?ๅ?ฐ่?ะ๋?๎??๐?ไ๒?ฆ๔?+๖?|๗?a O>8A?Nmฟr>?>์?}ฟปkฟ?~พg+นฝhฎ๋พฌ๊tฟH"ฟf]??ยื>า|ฟผ]?ฤฟฺเ>aฑ๙พ?4?G๕rฟดลq?ฝหพ+ฟๆ.}?๖พ฿sฟืฐ>ปu?ผชใฝห|ฟwษ?พ๕9?q?๛ฮ๙>ุฦพฮ]rฟ"eฟ๔Tพพ6B>ฮผM?	ต?๚`?bเ?ถฌ=๗=ณพ|	2ฟh?iฟmฟcอvฟ~Vฟ*(&ฟฌณูพq๘Aพu๒&=๎ซ>ไคแ>าu?ศ:?๘HT?าNg?*Vt?.|?ใง?์?|?C/w?ํ p?ีศg?^?#ฐT?ะJ?.@?โ5?4ป+?มั!?_7?๘?ฝ?
X๛>๐K๋>G?>ปอ>0ภ>]qณ>ฃwง>๊:>ชฒ>ึ>R8}>:๘k>aฺ[>ศอL>ย>>Nง1>n%>~	>mj>>/๘=th็=Qgื=ศ=Dบ=ฌฑญ=-ฉก=Cu=k=R=Lr=พa=๊R=ไC=U๒5=hR)==ฃ=ฺu=๚?<พY์< ๒?<ขญฬ<txพ<ฉ?ฑ<ร๑ค<x~<ึ<์<?ถz?๏'ฟฑภ>แ*ฟถ}?ห>Iฟg}ฟ๓~ฟ๔Ccฟ_พ}พE?ซ*?ล(hฟ&>ีแ?>aOฟE1f?Y~_ฟฃ๖5?VกพญOจพdk?u?Uฟฎ|พ;P}?บพฤ>pฟ>อi~?ต!>rV^ฟฯ Rฟ>๓ฎ=v_?ฌ๙k??ค>=โพpงmฟาtฟWฟ?Cฝะ๕>็๖X??ัฬo?
๔7?๘Bะ>O9=`พKปฟ7ภBฟกดgฟ~]{ฟษฟTงwฟSฬeฟ็Lฟ?V/ฟพฟc?พ^ศพ10พXTTฝN?x=Rี(>๛3>Syฑ>]ู>ร?>4z??5)?ภ&4?ฺ=?^F?ิM?G[T?Z?ศ_?u]c?ฤ#g?lj?รGm?5รo?2๋q?}สs?ju?}ำv?ขx?7y?ณz??ำz?ั{?1|?ฃ|?F}?z}?ะ}?k~?M\~?~?/ล~?Z๏~?เ?3?็N?ขf?/{?๛?d?ฝฉ?Mต?Pฟ?๛ว?}ฯ??ี???เ?ธไ?`่?๋?I๎?จ๐?ท๒?๔?	๖?_๗?๎n?ฝ๙5กพฆ=๛`F?Z-ฟฃฦwฟL้-ฟฟ๖คWฟJP{ฟ fnพ?3t?วV.=c]ฟณ{?z%OฟR๗-?hข4ฟๆ\?ฟ#}Y?สMพ2ฟศ}r?ํฟง<ณํ|ฟฦัT>?}?
P<tvฟฟ@	๙>%??Tร?Uเพkฟ๋๊lฟNฦ?พโr>ูD?ท~?kHf?ษ?@K>ูI?พอ3+ฟ$1fฟ??~ฟCฒxฟZ?Zฟณ+ฟ๚?ไพ๛Xพ*ภฅ<้]p>oู>p?์อ7?6R?ำe?ญ^s?จ{??xซ?ุเ|?ภw??ีp?ฮh?.i_?ุU?&rK?0!A?หา6?\ง,?tท"?:?อ?\้?ท??>ยป์>It?>ฯ>eม>Wด>?จ>(;>ถข>บถ>?~>๙m>7I]>?$N>@>gั2>&>ข>๓[>e>:๚=๏่=#ำุ=bาษ=ฤูป=nืฎ=ญบข=แs=r๔=ผ.=,t=?<c=ลyS=ฮD=&7=Nq*===]	=?จ?<T๊ํ<ฬf?<ฮ<Oปฟ<lฒ<^	ฆ<ซ<รศ<pอ<งิท>cFฟ๚r?ๆูฟ?ฮ!?_<?ีปพอ?;ฟทYKฟC๖	ฟn	C>4๗x?กค>ฤฟ5??:>ฐkฟัฮ;?Ig5ฟๆฑ?Uญชฝ	ฟ๋ฦz?ฝf7ฟคพB๒?ฬ%พ izฟ?>ท๚?ล> rOฟ:ญ_ฟญ๖ป7าS?3s?ฮตศ>&๘มพC๚fฟฯฑxฟช#ฟ}หฬฝจ฿>#R?๓}?ง!s?ฆT>?aเ>ษภ=ฆฺrพVฮฟ3t>ฟ6eฟ็;zฟ๒ฟุ๊xฟฤgฟjOฟึ02ฟบฟMแพผีพิ;พ๖"ฝ่๘O=nm>ฤ๊>ญ>โี>?๙>??c??	(?v23?ฺ=?_ขE?า/M?;หS?ืY? ^?ํ?b?ะf?B$j?เm?o?พปq?Jกs?มFu?xดv?บ๑w?โy?y๓y?Qยz?u{?|?ฐ|?b}?q}?ษ}??~?บV~?ญ~?ม~?ป๋~?ฝ?ห0?L?d?ky?s??จ?Oด?tพ?<ว?ุฮ?nี?#??เ?[ไ?่?E๋?๎?t๐?๒?X๔?่๕?B๗?EpN?๑8Xฟx??ฦ2ฟฏr?ฃ<ฝํClฟูyฟ}?oฟ"ปฟ|HGฟฦจ]>@~?๕|ฏพ?` ฟแ|?Hงsฟ?๕\?๖^ฟ+๖u?f๛|ฟBจ6?mE;I0Rฟร&`?ํS=>๖?ฟR/=?๕?mฃ>ฉีlฟgd+ฟ!ห>มด~?Z!?\hpพSTcฟzsฟgI๚พฝ\6>W;?w}?tk?ฎ๔?? ,>พฎ#$ฟิ@bฟR๙}ฟ`zฟPิ]ฟอ?/ฟ_m๏พ$nพอน}M]>ท(ั>?็	5?	P?่Ld?]r?๔{?รO?xษ??<}?Mx?q?mci?G`??V?aL?OB?6ย7?w-?L#?c๒?ข?ด?ค^?>๚*๎>ฮฯ?>ฉKะ>ฆย>ทต>แฉ>9;>>J>O@>	o>๐ท^>zO>AA>p๛3>|'>ผ>qM>F>ื?๛=ฌu๊=๏>ฺ=.%ห=?ฝ=-?ฏ=)ฬฃ=}r=wแ=_=หฦu=0ปd=?T=MF=ูZ8=2+=จ==aD
=ธซ =๊z๏<??<~cฯ<+?ภ<ณ<๘ ง<?<ๆบ<ลฎ<dฟโ	ฟำZ?|y7ฟkฝvบ?R ล>Z_พ>ฒพfบ;ฝfฑ ?.๎y?่ๅ๎ฝC~pฟSG?นhพ?พ฿F?๒๛พf๘>ฅล>ะ]3ฟด??ต#ฟฃR๗พ{?wบqฟy$<อ}?ใ_ย>ใ(>ฟ?jฟวฆอฝ๘F?Hุx?;o๋>Yu?พาS_ฟ`่{ฟ<x.ฟูพk๙ศ>ฐฦK??\|?ฆv?ItD?๏>บ >ธzUพb?พ:ฟาHbฟ๛xฟ??ฟ<๓yฟ ชiฟ??Qฟ??4ฟฟGฅ็พึ?คพqGพ๕ฝ/'=>&>y>ถฉ>cา>T?๖>?ณ?'(?(๓&?๊<2?
,<?ๅD?L?Z:S?YY?ะ+^?ะb?๐|f?y?i?ษl?Uo??q?ีws?ล"u?@v?ฆึw?fํx?฿y?ซฐz?Rf{?ย|?3|?n}?฿h}?ม}?~?Q~?า~?ฬผ~?่~??.?/J?b?คw?๊?ฝ?qง?Oณ?ฝ?}ฦ?2ฮ?฿ิ?ฆฺ?จ฿??ใ?ฟ็??๊?ะํ?@๐?\๒?0๔?ล๕?$๗?.?}ฝoฟrร|?~?ฟy/?@I?* ฟKmฟzฟ_cฟ้ืษพ;>?ม[?|-ฟฦพn"a?^?ฟ@๕x?ฌxฟ้?็ฑlฟ๗
?ฎัS>jฟ๋ฝF?ัAฐ>1}ฟ^
ฝH~?9>%`ฟฬ>ฟT2>7{?r2?พ	฿Yฟxฟ3_ฟ๑Y๒=๐<1?ฆz?ูp?$?ตgV>GWsพ?ฟฃ^ฟฃิ|ฟBึ{ฟฒ>aฟช4ฟL๚พฟพ?ฌชผa(J>ๅะศ>[?ๆ:2?๐M?Sผb?๔Rq?Ez?`?ํเ?}?อิx?4r?ฐ+j?0"a?UlW?OM??C?Wฐ8?|.?H$?ุฮ?v?7?ัเ?>๏>ิ*เ>Xั>nฯร>ูถ>ฏชช>;>e>พw>@>	p>&`>ไะP>B>j%5>]ฐ(>ฬ>็>>'>?=;?๋=ณช?=๔wฬ=ตPพ=่"ฑ=ฃ?ค=q=zฮ=?็=aw=9f=wAV=dG=9=ฏ,=ณ ==ฃ+=๑=๑<bPเ<lพะ<Aย<ลด<8จ<<	ญ<<>ฟท>P">฿e?; f:ฟEK?^]? ภ>9R>วE๋>}Dk?qH?_Pฟ3จ<ฟ?ฆs?น๓พุ;n>/?sพฅึ<ัร>?Vฟvz?Nฯพง\!ฟBZp?๐๛><ฟyษ์ฝcVw?/จ๗>'ฎ*ฟน๔sฟEพถเ7?เ|?๑j?0U|พ`ผVฟ3~ฟ&น8ฟํFPพ@ะฑ>!D?ZSz?ฆชx??PJ?๒ว?>ข >v๋7พ
Q๓พ๋}5ฟฺa_ฟKwฟฦ๑ฟ1๖zฟป|kฟDTฟ+ฤ7ฟ~ฟตถํพq?ชพฮSพฌฝ_>?<B>๛กp>ะฅ>๒แฮ>zบ๓>N
?A๋??%?F1?S;?ล&D?oใK?ฃจR?X?ืผ]?=b?ม(f?<i?์l?Ao?๎[q?Ns??t?ีuv?eปw?ฤีx?สy?็z?์V{?j๖{?ฃ|?i๘|?0`}?บ}??~?zK~?๎~?ธ~?kไ~?h
?N+?อG?{`??u?]?e?Hฆ?Nฒ?ทผ?ปล?อ?Nิ?)ฺ?;฿??ใ?m็?ธ๊?ํ?๐?.๒?๔?ฃ๕?๗?`_ฟ3ีพ3ญB?0ฟNฎ;รy?ภฆ๛=ฐฟฒ๑7ฟเ'
ฟ9ง=ํf?ขม?ห	hฟฃMZ=ค+?Y๔rฟH?ฺฟณUy?ๅ(Oฟีฑ>Mิอ>ฺyฟh'???>}	tฟS`พx?ศFภ>LPฟ(Pฟ6lS>๚u?ำYA?ธฝ?0Oฟๆ{ฟKฟึ/n=[&?w?ๆt?Zร,?6>!Lพึ]ฟฯYฟ<o{ฟฎ}ฟึ}dฟ9ฟNฟ?์พhh)ฝ_๐6>ัhภ>%?a/?XฝK?k!a?อ>p?	แy?sู~?ิ๑?nไ}?ฦWy??r?๐j?}๚a?QX?ถ:N?ึ๏C?-9?e/?gc%?ช?YI?rI	?ฑ ?๑>Zแ>กฺา>โล>ด๛ท>Gปซ>ู:?>r>X>ไ>Xr>a>'R>ุฟC>TO6>1ฦ)>ั>T0>	>T$?=ยํ=p?=ดสอ='ฟ=Hฒ=๏ฅ=ญo=zป=ฤ=A?x=อทg=MฅW=ฒฏH=Zร:=๙อ-=พ!==ๆ=)Z=๒<-ลแ<Yา<แร<x๑ต<+Pฉ<B<,<oq<6m๙พฐhv?q>&ฟแ9??ฟฌ`>]~?EเU?่2?0W?,%?Y??>#ยQฟชGุพ?ข?c๏=ฟด[ก>)@rฝd@=ใ$hพ๛f?ื3pฟgj?มัdพ"๙AฟM^?ฏ>ฬyฟ?|พ+Cm?๛ั?V;ฟํ{zฟษพฏศ'?ผC?ฺ\?>6พ1=MฟฟdBฟ7พ(;>D๊<?็ืw?ก?z?c่O???.@>p3พ?ใๆพ&ี0ฟปZ\ฟvฟ์วฟฐแ{ฟE<mฟGVฟ.}:ฟ๖ ฟRฝ๓พPืฐพZ^พ?๓มฝ๊Yช<d!>1h>i็ก>`]ห>๐>ม็?๎ฌ?ฒม$?N0? y:?kgC?ฺ;K?R?$X?6M]?ุ?a?ิe?Hi??Il?ๆn?+q?$s?ฺt?7Vv?๘w?๛ฝx??ตy?z?mG{??่{??t|?S๎|?sW}?ฒ}?p~?หE~?~?Mด~?บเ~?5?(?gE?g^?t?ฮ??ฅ?Jฑ?ืป?๙ฤ?โฬ?ผำ?ซู?ฮ??Aใ?็?q๊?Uํ?ึ๏? ๒?แ๓?๕?้๖?ฃผaฟ$?ข?บ4C=ลz-ฟ๊wc?ชฎ2?Y๔ท<-ยrพ
h?ฝ"?	?t??-@>qมฟ๎ะ>?ล>ZัMฟำ,p?๗sฟ.งb?&ฟ%@>ฦ?ูฟ ?s!?:Ieฟ์4ถพ?}m?Q๙>ทร=ฟuฯ^ฟ๒ญ?=เยm?@#O?ฉ8;TWCฟx~ฟC&ฟ-ป๔b?nึs?สw?ว$5???>V$พ)ญฟฉใTฟxษyฟ ~ฟgฟ=ฟั~ฟซภพh}ฝCง#>*๑ท>& ?|,?`I?D|_?-!o?F8y??~?.??ถ/~?๖ีy?s?ฒk?๘ฯb?3Y?c$O?:?D?ถ:?rM0?งE&?ค??5
?rq?๓t๒>a฿โ>!ิ>8ฦ>กน>จหฌ>f:ก>a>P8>?ต>ขs>ec>)}S>?D>.y7>๘?*>ฬ>น!>้	>มc >?	๏='?=nฯ=วภ=Snณ= ง=An=yจ=<ก=๘z=6i="	Y=โ๚I=๗;=?์.=ษ"=}='๚=a1=ฉ,๔<๗9ใ<Ftำ<ปฦฤ<์ท<ลgช<t<O<รR<i|๑>0?J?เ?ฟต?ไA<ฟๅ๊พS7?z๏?มณx?hธ?dดW??แปs{ฟ์2ฝQ่i??Hlฟว;?เ8ฑพ#>O๎พ??B?ึม}ฟ๖}P?+Qฝrw\ฟปซF?ไทใ>ง=oฟl#ฟพะ_?dิ+??พm~ฟฒหฝพมn?A??gv%?ตฃ฿ฝเBฟึ?ฟฆrKฟM๐พsH>ๅ4?ฅ๋t?บซ|?๐8U?--?;`>vฒ๘ฝJฺพ๛,ฟใ3Yฟ<tฟvฟฃต|ฟ่nฟ?โXฟเ*=ฟI่ฟีธ๙พ6สถพLjพพูืฝ2โ0<<Y๓=๔[_>
?>แีว>ฒkํ>i?0m?๖ฆ#?ฯT/?ว9?งB?WJ?นQ?oW?๎?\??ya??~e?k?h?f	l?ฎn?๊๚p?เ๙r?sตt?f6v?_w?ฆx?>กy?{z?ิ7{?w?{?Hi|?-ไ|?จN}?้ช}?ึ๚}?@~?|~?ฐ~??~???ภ%??B?P\?>r?=?ฐ?๏ฃ?Fฐ?๕บ?6ฤ?9ฬ?)ำ?,ู?`??โโ?ษๆ?*๊?ํ??๏?า๑?ธ๓?^๕?ห๖?Y๊ขฝ฿๗?ิRCฟพ๋@?)Aฟฦงล>^+}?&?)ข>?๊>ญว^?฿be?>Xlพdpฟฮด4? b=.EฟP&L?ผๅUฟU=?6็พ?iดฝ]๑;?ฏ?|ฟ"ฒ>l๘??ษQฟ๓ซ๘พSญ^?ณy?ฟณ(ฟทkฟx<๊c?G_[?ฎ=Ue6ฟฬฟN2ฟoFฝฬด?๑lo?jsz??&=?ฉ>จ๙ฝฬฟ๏Oฟรใwฟ?่~ฟ๒wjฟ,แAฟฟnขพ7ฆจฝฺN>jฏ>d?ง)?ะ6G?๐ฬ]?!๚m? x? E~?๛??[u~?YOz?#t?5pl?ขc?Z?!P?วE?๐r;?N41?''?๙_?K๎??
?e1?ดแ๓>็8ไ>๙gี>ะkว>J?บ>ั?ญ>ว9ข>โP>m>>*u>ฅqd>&ำT>M>F>๘ข8>ฒ๑+>ฝ >>vส
>S5>ด๐=ึํ฿=#pะ=?ย=ด=?จ=ำl=v=ุ}=ญ1|=cดj=๕lZ=FK=ึ+==ฝ0=ิ#=v=iแ==<ฝ๕<มฎไ<3ฯิ<	ฦ<^Jธ<^ซ<ฅ<r<4<Q0?	ผธ{%ฟ6G(?๖-ฝก'lฟ๎ฺ>๔WO?รr?zc?;?>Pใพyฟ๘๒ฎ>๕T5?*ฟๆฐP?vฟฟง?BN,ฟ๔hd?9ฟ?ั-?๐>ฐpฟ\)?Iฎ?,ว_ฟ?พZ`N??@?ะ฿สพ์?ฟU)้พ7๔?o??ก3?~L"ฝฑ7ฟSฟ?Sฟห;ตพVT>?,?่q?;~?ต@Z?ัS?ต>๋ฦผฝVอพ6,'ฟฦํUฟcรrฟj!ฟ๖q}ฟpฟ[ฟอ?ฟNว ฟ๛จ?พ็ตผพ@uพนํฝภP:Kjเ=lฒV>>Kฤ>ุ?๊>?	,?็"?NZ.?nม8?คๅA?่้I?๎P??W??k\?a?')e?ืณh?ศk?3vn?๕ษp?]ฯr?t?bv?hw?๕x?[y?ํhz?"({??อ{?]|?๖ู|?ฮE}?>ฃ}?2๔}?R:~?w~?ฑซ~?Eู~?ภ ?๑"?@?5Z?lp?ฉ?R?ภข??ฏ?บ?qร?ห?า?ฌุ?๑??โ?vๆ?โ้?ู์?j๏?ฃ๑?๓?;๕?ฌ๖?!ปK?โฌ?o|ฟ'?~?OHฟๆพb`?๊]q?2-F?_eW?	ๆ?Zy?*?ฟฮู<ฟ์i??รพฃ	พค?ภใ'ฟศ?ฺjพตคพ&p[?0ฉpฟำ4>?Y?j๛7ฟ๙ฟp่K??ฺ/?dฟษtฟฉ9ผฝ]X?o๖e?ฤ#)>ใl(ฟ8โฟzู=ฟ{๛ฝU?ฆ[j?ผ|?วลD?๖บฝ>ํทฉฝำz๛พ
พJฟพuฟ{~ฟฤ1mฟFฟ๚ฃฟ๏-ญพ-าฝ่ั๙=เีฆ>ฯฟ ?j&?ศใD?\?ฒษl??อw?๏}?9??]ต~?๏รz?ๅฟt?์*m?prd?R๑Z?์๑P?7ฐF?ฺ[<?2?(?9?ๅฟ?Pฅ?๔๐?ึM๕>์ๅ>	ฎึ>Gศ>ฏ`ป>ร๋ฎ>?8ฃ>@>m๘>Y>dฒv>ล฿e>)V>i}G>ฒฬ9>_->ฃ!>j>\ซ>฿> ๒=~Yแ=ายั=b>ร=ฐนต=k#ฉ=ck=q=rZ=_ฬ}=ช2l=ฦะ[=>L=`>=*1=฿$=n=ฉศ=ฯ฿=ฯM๗<#ๆ<*ึ<oLว<ัvน<๗ฌ<ื?<u<l<เ?ฏOฟไA&>๒#ตฝฏ?ttฟ๙ด๕พwช>C"?vY
?5ๆ<ีJฟKฟXี,?~ฏั>sณuฟๆst?+	OฟRBA?ใwVฟ-y?โsฟ'ุ?๑ฎ>๚{ฟ?l??2?KบKฟAศฟZ
:?นดR?็Hพ?๊~ฟ:A	ฟ	๙เ>ฯ{|?ฺส@?\๑๖<Yผ+ฟ~ฟ
[ฟCฮพ	#>๓ม#?6ฦm??๛?^?RN?Jw>ฑฝภพง-"ฟ฿RฟF้pฟิค~ฟ~ฟ"rฟซD]ฟงcBฟ฿#ฟพฦฟ(ยพ้พธศพuสผavอ=ฦN>}>Nพภ>็>Xช?~้?m!?^-?๖ใ7?8#A??I?YP?ฮV?i๚[?ด`?๏าd?ะhh?Mk?=n?ณp?คr?mkt?*๖u?ฆLw?ธux?Wwy?ดVz?W{?-ภ{?ขQ|?ฎฯ|?ๆ<}?}?ํ}?4~?r~?Yง~?ี~?}?~? ?>?X?n??๒?ก?7ฎ?-น?ซย?ใส?า?+ุ???!โ?"ๆ?้?์?4๏?t๑?g๓?๕?๖?]p?Iพฒ ฟf@ ?UพRWฟ/ั>w?๊~??ต?e?็O>@Uaฟฝฬุพe?ซฟญ]q<๗จ>ชฝฺพ]iข>รป;TQฟQถq?ธณ[ฟ?กน?l?Xฟl6ฟจ5?จE?์J๐พE{ฟopCพ๔gJ?fิn?7ใy>ฟ
น~ฟ?Hฟูโ:พฤ
๎>ฆd?>k~?%?K?ั>็3ฝ๋พPEฟZsฟU?ฟพoฟํ$JฟxฟูฤทพjO?ฝย๎า=?3>๒#๚>#?eB?PZ?๎k?w?}}?๋๓?น๏~?ด3{?ฒXu?8โm?h?e?ฏฬ[?ฤีQ?อG?qC=?ห?2?#็(?|?่?งm?!ฐ?Wน๖>n๊ๆ>ฏ๓ื>iาษ>ฮผ>~๛ฏ>8ค>*/>Pุ>*>:x>วMg>ึ~W>rผH>\๖:> .>!">ต๕>;>fุ>๓=ลโ={ำ=มyฤ=X฿ถ=ึ4ช=๐i=jo=7=g=๐ฐm=4]=j?M=M?=}I2=~๊%=g=๊ฏ=ท=b?๘<S็<
ื<Iศ<Dฃบ<ฎญ<	?ก<ทg<ภ๖<z[ฏพขtฟฤW[?fฃGฟ่9}?ฝv
ฟตจiฟเพW๛=คC= ยโพซzฟq?๒พผ๊g?๋oh=ูฐOฟใ๘?ฯqฟ๘ug?jวrฟํ??ฟ๏\ฟ๙งจ>ng???ฟึร>๓L??3ฟไ4ฟโื"?b?bๆCพJK{ฟปฝฟZธ>ฎBx?m?L??Qฬ=ฯฟNณ{ฟFฆbฟTDๆพฒCใ=๗ฌ?Ki?^ม?"oc?#?๏>;?ฝๆณพ#ฟจOฟ"๑nฟภ~ฟzฃ~ฟysฟP^_ฟk๎Dฟืk&ฟณฟผvศพฯ?พฑพNผ้}บ=+SE>p*>U.ฝ>m฿ใ>จ=?ฅ?ุN ?ฃa,?`7?ษ_@?IH?ฉรO?ไV?-[??P`?5|d?Xh?ซEk?sn?$gp?yr?Ft?ภีu?0w?U]x?1by?_Dz?r{?gฒ{?ฒE|?Vล|?๐3}?ร}?ศๆ}?ด.~?m~?๙ข~?ธั~?5๚~?G?ง;?๕U?พl?y??\??-ญ?Gธ?ใม?6ส?lั?ชื???ภแ?ฮๅ?Q้?[์??๎?E๑?>๓?๔๔?o๖?ฑ`>ทืqฟ์ฃ>+0พ\?๕}ฟ็๔gพM%?ณg?c?ษ?
|พhRฟ<7ฝ์s?VRฟ `ฆ>ื$=อ*พ`=3Bv>ถ.ฟBิ}?พ>ฟแr5พขvy?aบ๒พ๙INฟย็?พ*X?qบพ4ฟhพ)๗:?)่u?ผค>gบ	ฟqR|ฟ
~RฟAYwพ60ิ>LP^?	?JษR?งๅ>?ปDฺพฤจ?ฟ.ธpฟx?ฟRrฟฝNฟIuฟ?Dยพต?พไ๗ซ=>พถ๒>ษ ?ษ@?ฑX?แLj?a@v?๛-}?ไ?n$?ง{?lํu?n?	f?ฌฅ\?ฅทR?ล}H?ต)>?iโ3??ล)?จ๊?Ta?5?๊n?6$๘>mB่>๊8ู>6ห>จขฝ> ฑ>?6ฅ>>ธ>๘๛>ญมy>ชปh>ิX>f๛I>๕<>2/>P$#>๘ๆ>m>็ฉ>?"๕=น0ไ=hิ=ตล=?ธ==Fซ=zh=b\=ข=?=3/o=d^='O=ศ@=\h3={๕&=_=*=<=๔n๚<้<๖฿ุ<"าษ<ถฯป<)ฦฎ<:คข<ูY<ุ<ุยyฟ่งพฅฉr?ง?}ฟ๊R?๙
>}XyฟHzCฟdธูพ_t๊พฯOฟxฟ?๗ฝ2พ?ihพCํฟjr?่สฟๅi|?ผhฟญ|x??E;ฟ?>ณฝ*?๎๒{ฟ์e>Qga?,ฟ๒KฟF$	?dn?ว๐ญฝ,)uฟ฿.ฟVY>mr?6หW?๔,>?ณฟYkxฟ๛hฟ฿?พกึ=ฎF?๐d?X??จg?ดถ)?0>ฎ>=่ปบZฆพเฟคdKฟ4?lฟ@Y}ฟฟJืtฟLhaฟ;mGฟ1)ฟBฟjKฮพ๒พ8พึ3๏ผNง=ฦ<>๔4>ขน>๏ชเ>ฯ?D`??.?}c+?ฎ%6?W??่G??,O??U?J[?ิ์_?๚$d?mัg?คk?หm?H5p?CNr?} t?#ตu?<w?สDx?้Ly?์1z?t๘z?ค{?ฏ9|?ํบ|?๋*}?๓}?เ}?ื(~?ํg~?~?้อ~?่๖~?k?-9?ะS?โj??~?+?'?"ฌ?_ท?ม?ษ?ีะ?(ื?ก??_แ?zๅ?้?์?ฦ๎?๑?๓?ะ๔?P๖??ะ3ฟOฬSฟ๔ญm?Nฟ๏๋y?!?-ฟธ๖Dฟ0ญว=๛	?
?ส่>ใH%ฟpฟ์hฎ>่H?ป๒vฟ?Z?ถฉพ5ู=?>พแn์>MSฟG?ลืฟๅmฒพ.?sตชพ้าaฟV?>ฟzg?งkพ๏ฟ?รพ1่)?%${? ห>Y๒พjฑxฟ๎[ฟ?tพ๎ฉน>ฎ^W?ม๙?ฅ&Y?7๘>`Y=6ษพQศ9ฟ3ุmฟ?๊ฟLtฟKโQฟน< ฟฅฌฬพึล'พ๒๐=iษ>8๋>รl?ญ=?uซV? i?Smu??ม|?ฅอ?{S?ล|?~v?Fo?ระf?G|]?S?bI?ค??๏ฤ4?ฒฃ*?ย ?)1?ไ??N-?r๙>่้>ป}ฺ>ฌ7ฬ><รพ>Jฒ>5ฆ>เ>ผ>Dอ>I{>n)j>"*Z>E:K>~I=>H0>'$>2ุ>ๅM>b{>.ฉ๖=Kๅ=ผบี=r๐ฦ=*น=ขWฌ=g?=WI=8๐=3N=tญp=0?_=พrP=ม?A=:4=w (=๛W=i~=re=?๛<ใ๊<แ:ฺ<๛ห<(?ผ<ม?ฏ<kจฃ<?K<hน<฿66ฟฬ?b4พ>๕ฟBั]>็;?#ฟฤว~ฟ้กWฟEWฟ}ฟู}Cฟ8ฏ>qฐp?ฐฟ)๒พz;L??ศwฟ^~?{ฟ๔c?ืฟwฝจ?K?๓oฟๅ?y=Zq?)๑พ๚]ฟpฉฺ>(w?ทณท<๘lฟ็|?ฟF>งk?}a?่r>4ผฟ;;tฟPnฟแc
ฟW*b<	?็_?iุ?!gk? 0?ฑ`ฝ>งะ<พกฟYฆGฟภงjฟf|ฟศuฟฒ!vฟ{bcฟ๐฿Iฟbํ+ฟAyฟ๖ิพฉพ้t"พ ฝ?=รไ3>=>Cถ>คs?>` ???$d*?แD5?โี>?;G?N?เU?ยกZ?_?=อc?g?9มj?Tm? p?ด"r?ญ๚s?Su?ล๗v?,x?7y?\z?]่z?{?-|?tฐ|?ุ!}?}?4ู}?๐"~?ัb~?%~?ส~?๓~??ฏ6?จQ?i??}?ฤ?๑?ซ?vถ?Qภ?ฺศ?>ะ?คึ?/???เ?$ๅ?พ่??๋?๎?ๅ๐?์๒?ฌ๔?1๖?น{zฟ@3ฝqob?K|ฟฎญ\?บGฝ์มฟ?q๘พุแ;4รF=ฃฏพฃ0jฟ ฎ7ฟ&,???๖Pฟ6R?Doฟ๕ฝ>?ฺพt@(?าฯmฟฬ v?โพุ@ฟ~?{L=พธบpฟฬมม>a;s?@พ๕ื}ฟKแ๑พิ`?N~~?
.๐>?เฯพ}ฺsฟ๐cฟtฌถพb>ึO?ุ??_?ฑo?k=lใทพ๒ฐ3ฟWปjฟฟMvฟ๖Uฟํ$ฟ์๚ึพ?z<พ%ป;=ฺ>ฯฉใ>/L?e1;?rสT?ซg?ไt?N|?ฏฐ?฿|?f|?
w?๓o? g?P^?}uT?ีDJ?<๒??]ฆ5?ข+?ะ!?e ?ษร?N๋?๘๚>฿๐๊>!ย?>หiอ>ใฟ>[)ณ>4ง>๛>Ew>w>[ะ|>k>ข[>yL>๖r>>]1>า)%>dษ>ฐ.>ุL>v/๘=ึ็=Tื=ฤ+ศ=;Pบ=iญ=eก=J6=ฬฬ==ด+r=๚_a=ๆฝQ=๙0C=ฆ5=r)=tP=จe=จ<=?<ช๖๋<ห?<ำWฬ<(พ<Z๕ฐ<ฌค<><ผ<pS>7ม?!?๎พฬ->๑มฟฒ?ฉ=2ฝDื_ฟr?ฟหฒฟ็pฟหฮพพQ2?Y=?ิ\ฟ(=?yZฟบm?0ngฟข๑@?ลฝพญฎพe??b\ฟ'?ีฝt{?#0ฎพ	๖lฟlญ>k}?ก>กกaฟrNฟ??=ภb?G๓i?้ึ>>m๊พ?&oฟxjsฟBwฟวใฝB4๛>DyZ?ขL?A๋n?zV6?ธRฬ>w`=$พc.ฟQหCฟWhฟJ?{ฟ!ปฟ,XwฟนLeฟcFLฟซ?.ฟ็Rฟ&?ูพ:HพฬO-พ๚tIฝ`}=K(+>๗B>Fnฒ>9ฺ>P??>ั?๋?c)?๙b4?m>?F?9?M?ฦ~T?-Z?อ"_??tc?A8g?k~j?AWm?ซะo?ใ๖q?ฃิs?Psu?!?v?Bx?๖!y?ฏz?,ุz?{?o!|?๊ฅ|?ถ}?-|}?Yา}? ~?ฌ]~?ฑ~?7ฦ~?>๐~?ฆ?-4?|O?"g?{?[?ธ?ช?ต?ฟ?*ศ?ฆฯ?!ึ?ฝ??เ?ฯไ?t่?๋?W๎?ต๐?ย๒?๔?๖?ษถตพนiF?ม๓^>hฟ>่ป?ธoRฟ?geฟฤ?ฟ๊พภ์=ฟ5าฟช3ฝพหg?9*>ศ{jฟำ9u?ไ?ฟๅบ?d=$ฟง?P??ษ|ฟcb?Gพ/'ฟบv?ฬฝ,ณzฟ>_>{?ป๕ูผฒาxฟ?ฟ?7๐?>ๆ	?=ฌพฐำmฟWทjฟ๑9ำพ๐>แปG?.?ิd?$?งm็=?Qฆพvd-ฟdbgฟฟxฟ Yฟน)ฟf.แพQพด?<8cv>#?>="?฿ซ8?ภ฿R?Lf?ฎs?ำ{?.???|ย|?	w?p?Vh?P"_?oQU?้%K?{ิ@?ฐ6?ซ\,?ษn"?ฯ?1?้จ??`?>QG์>?>ฮ>ม>38ด>V2จ>๊>ฐV>o>wW~>m>ี\>วทM>^?>?r2>,&>บ>s>H>ตต๙=Zs่=ๆ_ุ=gษ=ิuป=azฎ=dข=;#=^ฉ=ู่=๑ฉs=รรb=	S=0eD=๓ฤ6=m*=ํH=ๆL=?	=ฆ ?<qkํ<ถ๐?<ฌอ<Uฟ<๒ฒ<อฐฅ<@0<|<ขUo?#ฤ!?ว?yฟฏT?[uฟฯ๑K?Nศ?^<ใพYฟฐcฟง+ฟ?=)แm?พQู>q|ฟิuอ>๙>a*ฟ]/K?ห^Dฟณ?ญ!พ๏พ็v?ท?Aฟฅพ??(JOพํผwฟtyD>อ่?าฤp>'nTฟ๕[ฟษณ<5ฏW?nq?]wฝ>eฬพ?2iฟ|wฟแ! ฟต?ชฝjผๆ>xงT?:\~?าr?oV<?r?>TZฌ=บ|พญฑฟิ?ฟ`้eฟzฟ่ฟกzxฟๅ&gฟp?NฟฦJ1ฟ&ฟพ฿พ?๋พ%8พK[rฝสํ\=h">F>ทำฎ>ั?ึ>ส๙๚>D?:ว??a(?๚3?๘G=??E?dM?๓๘S?มธY?๏ผ^?@c?๋f?8;j?ึm?้o?ฮสq?^ฎs?Ru?Qพv?E๚w?Jy?ๅ๙y?ใวz?qz{?3|?P|?}?7t}?tห}?~?X~?5~?Uย~?แ์~?ผ?ง1?LM?>e?๚y?๐?}?๕จ?ด?นพ?yว?ฯ?ี?J??6เ?yไ?)่?[๋?๎?๐?๒?d๔?๒๕?M??Ax?5ฟ%ฑV>5ใพ๋y?`ๆพU}ฟฟ-fฟ%Wฟ๏+xฟลฯaฟุ0=?บ?9บIพ&ี:ฟf๋?PEhฟเL?เoPฟ
ฬm?ฟ๛BE?1ฝjืFฟุh?X๙=๒ฟา๘=-d?sรต=๎pฟ๖Z#ฟ ?>w?Wะ?งกพคfฟrฺpฟ๏พZำM>ฉ??iร}?ฎi?ธ]?>ฦพนไ&ฟ7ฮcฟ๗X~ฟภyฟ5w\ฟ๒.ฟอE๋พกฅeพ๚;ษฌd>]ิ>๏?ค6?t๋P?ิไd?ยr?=Q{?"c?ฉพ?}?Zx?%Cq?0i?ธ๑_?b+V?XL?aตA?็e7?ฬ7-?D#??P?f?Nษ?><ํ>จI?>อฯ>Q#ย>ัFต>x0ฉ>Zุ>?5>@>k?>๚qn>T*^>i๖N>ตล@>Y3>(/'>ญซ>/๐>ฒ๏>๊;๛=ึ?้=qฒู=Xขส=jผ=ผฏ=bฃ=*=๏=*ถ=,(u='d=2TT=fE=ฮใ7=g!+=eA=$4=๋	=X =7เ๎<K?<?ฮ<|ภ<$ณ<?ดฆ<a"<c]<;ฤM?$๐yพsWLฟkOz?Teฟ๑`c>๕[s?hd>Qเพผ
ฟบK{พั4๑>ไย?เ<=ป๛zฟX/??dอผGืพํ??ฟฝฝ>K~=+#ฟWA?;=!ฟ๚ืพu~?ถะwฝฏ~ฟ	%=a?~?ญ>OEฟฬไfฟNฝMโK?ฑฺv?ธ+?>]ญพ.ebฟyฦzฟE\*ฟ฿พJหั>แtN?}?ฟ?t?นB?้>คR่=ลุ`พiฟCม;ฟ_cฟฏzyฟ?ฟ๘yฟ?๐hฟ๑ํPฟ๋3ฟก๒ฟJๅพฌขพโ๕Bพดฝ๏?6=ซฅ>|>ค6ซ>]ฝำ>๘>=?&ข?๓^'?ใ2?<?O.E?4สL?frS?ICY?~V^??ยb?Of?ข๗i?โl??jo?wq?เs?ฑ0u?Uกv? แw?|๖x??ๆy?ทz?=l{?ไ|?ฅ|?H}?5l}?ฤ}?~?JS~?ณ~?mพ~?้~?ฮ?/?K?Vc?Tx??A?ใง?ฒณ?์ฝ?วฦ?rฮ?ี?ืฺ?า฿?"ไ??็?๋?็ํ?T๐?n๒??๔?ำ๕?คo?bฤ>uฟ ฟY?oฟ/ษc?6ยง>f;ฟญx~ฟ่ฏฟดธxฟ 8ฟ*ๅ>jษp?Qผฟ|?๋พู;q?ฆ}ฟชภn?๛+oฟ!j}?ีิuฟSุ?฿๊=M:`ฟ7&S?0>ฎฟ๑?Cผw?BฬO>ฺBfฟ76ฟ?Aฑ>?}?7น*?+Dพ๒U^ฟข๙uฟ_๕ฟ>้5?ั{?ษn?e๏?๏ C>?พง3 ฟบ?_ฟb}ฟฆ2{ฟฃต_ฟm2ฟแ?๕พ๐zพ<ผืใR>?ฬ>ณ?ื3?ฆํN?tc?อq?qวz?2?ื?หl}?x?ภๅq??ะi?ตพ`?VW? ใL?๊B?D8?.?$?}j??์"?๗0?>?๒๎>ว฿>?ะ>ษBร>5Uถ>k.ช>ฦ>+>~>ฒ>>฿o>_>๕4P>๛๎A>ฉ4>ร1(>ล>ๅะ>ม>ย?=KJ๋=๗?=?ห=๛ภฝ=ฐ=
aค=?=~b=z=eฆv=Pe=VU=อF=ฉ9=`,,=?9 =b=Fย
=โ =?T๐<ฆ฿<[ ะ<ํญม<!<ด<.นง<<ท><ฉ?ฝiklฟ1เnฝะ?ดพธฉ้พG?q?9C.?y฿=นpJฝ
qr>โJO?แิd?า?ญพlXฟGc?ฟ_ซพTพ'ภธ>กถพH*>ก?>ฤ๖GฟคP~?ร๗พ|ฟฟืNv?฿Mฉ=Q๛ฟฉaฝqฌz?ฟ฿>Kะ3ฟ%+pฟิHพฃย>?ภ={?8ห?>fเพสรZฟ]E}ฟB4ฟ้P8พๅkผ>PไG?&O{?w??ญG?๓฿๗>บ>จhEพ๋๘พg7ฟ_ธ`ฟm?xฟฏ๛ฟzฟชjฟฤ.Sฟใ6ฟtธฟE๔๊พ*"จพqภMพ"
ขฝฦ=ู฿>ะt>ง>E{ะ>ฅ(๕>ข๑
?ั{??Z&?ทถ1?ถ;?}D?{/L?!๋R?-อX?{๏]??ib?,Of?จณi?๛ฆl?7o??qq?&as?u?,v?ีวw?เx?๙ำy?งz?๓]{??{?๊|?๛?|?&d}?ฝ}?๘
~?N~?)~?บ~?ๆ~???,?ใH?la?ซv???ฯฆ?รฒ?ฝ?ฦ?ือ?ิ?bฺ?m฿?หใ?็?ู๊?ฎํ?#๐?C๒?๔?ณ๕??q๗>๑ฟ2๔0ฟ4่w?าlฟC๏ฦ>	U?t;]พ:dHฟ๗สcฟSr?ฟ61พ?`F?ิ<=?ัMฟ|Gพ?J?๒|ฟtึ~?u~ฟ$ู~?แ;`ฟCl็>ฃ/>พrฟปk8?ภtี>|yฟEพ`ๆ{?๙?>๏XฟBGฟC?>หx?m9?\๑ฝO๒Tฟ^zฟyํฟFไท=?=,??Fy?ศr?F8(?kjj>ศย`พ:Sฟๆ๗[ฟ๏2|ฟyt|ฟโฮbฟบ6ฟb?พ?6พ`๚ผำ	A>ปิฤ>!n?แ0?lๆL?q๚a?๙ะp?;6z?n๛~?ล้?ฆบ}?y?฿r??j?Fa?FูW?@ฟM?sC?? 9?Q๋.?C์$?N7?vฺ?S฿??K ?}G๐>yฯเ>ื.า>๙aฤ>_cท>.,ซ>ด>;๔>Oโ>๎u>bLq>ิ`>msQ>0C>๋ฒ5>S4)>ำ>ฑ>v>9H?=ธต์=wW?=ูอ=ๆพ=iฎฑ=_ฅ=๊=?=ษP=$x=๏f=y๊V=ฮH=!:=Y7-=T2!==z=)้=ยษ๑<rแ<3cั<]ฺย<นSต<^ฝจ<ฅ<
 <`ฟY๚[ฟื?8?pพ๑nย>ุโkฟค๊?<๔y?O?ลข้>๔)?'#|?ฯ!?ๆh,ฟะRฟB}?>uฟTช>n๏ย=^เฝฺยฝ๗>\[dฟธt?cาฃพ}1ฟ้ฐh?หe>YM}ฟ๓6พณs?ุ์?Wน ฟt]wฟbBqพf0?J7~?;?จN[พ๔URฟG๗~ฟ d=ฟ!iพvฉฆ>ฤ๘@?ฏ3y?ุมy?M?.๕?ยอ/>ฅา)พูoํพK3ฟฏ๕]ฟ^้vฟWแฟ๙h{ฟฎSlฟฤbUฟ9ฟjwฟภ๐พ์ดญพ๊XพjrถฝหYี<?>wl>%๕ฃ>6อ><๒>\ค	?>T?U%?wะ0?ฆ๋:?ฬC?๔K?#cR?mVX?็]??b? f?Koi?kl?ฺo? Eq?3:s?Hํt?ืfv?dฎw?}สx?ืภy?mz?O{?๐{?{|??๓|?\}?ถ}?ใ~?ศH~?~?ถ~?ซโ~?ใ??)?ชF?_? u???ย?บฅ?ำฑ?Mผ?_ล?;อ?	ิ?ํู?฿?sใ?F็?๊?uํ?๒๏?๒?๖๓?๕?P{๓พข~ฟwา=ื6๛>คึพตพ๓?]ม>~0ฉพ๎
ฟDSณพ>ณ:w?ฎึู>/>vฟฺL>์?:คfฟ๎{?ฝO}ฟr??ย?ฟTn>๔>฿4}ฟWง?(?
ผnฟีพPt?%็ื>ฅIฟ์NVฟ฿?*>ฅr?cG?ปข0ฝPJฟ@}ฟ2]ฟ๋	="?%v?u?40?ภซ>#<พzEฟรทWฟ๎หzฟึ}ฟlยeฟฬํ:ฟkฟ/Sพ{Kฝ. />๛ผ>จ 
?6.??ีJ??w`?ฬo??y?ศฝ~?ั๖?ฃ~?y? s?w?k?gQb?2ญX?ตN?ไOD?ฺ?9?ณร/?Bฟ%??ไ?R	?)? ?า๑>ฝโ><_ำ>แล>Oqธ>ร)ฌ>ข?>,ำ>ณ>,9>fนr>)b>ะฑR>TAD> ศ6>ุ6*>ู~>:>ฯc	>Sฮ?=!๎=๐ฉ?=Tฮ=ภ=ปฟฒ=?]ฆ=๋ึ===ะขy=ึRh=5X= 6I=[@;=QB.=ห*"=?้=ฎp=pฑ=>๓<Z\โ<
ฆา<ฮฤ<Pkถ<มฉ<ฦ๘<^<F3aฟ1ำฝ'ฆ~?"_ฟhh?จtฟ๊|ฝแm?vq?งW?ึศo?ศv?๛?>gฌgฟพ4ำz?ถSฟ๓,?>Uิ5พ>๐ไฆพ6)?e'wฟa??พ?Mฟ+U?ุธ>๔vฟz์พB h?}h?Fฟศk|ฟ6ฃพQไ ?	ร?ส?ุ๏พ#Iฟืฺฟ$Fฟเฑพo>aต9??ตv?dค{?สR??ุ	?าhM>	พ4อแพ้.ฟT[ฟจxuฟฏฟ{:|ฟL์mฟัWฟ?;ฟd/ฟฟ+๖พฟAณพ?Bcพึสฝา"<?=/d>ีP?>T๏ษ>๊L๏>อU?o+?3O$?$้/?< :?C??๗J?nฺQ?	฿W?ย]?<ดa?ฑe?*i?ร/l?็ฯn?แq?s?Gหt?VIv?ฬw?Kดx?ญy?พz?A{?ใ{?Ap|?7๊|?ไS}?rฏ}?ฤ?}?{C~?~?ฒ~?9฿~?็?h'?mD?]?Rs?,??ขค?แฐ?{ป?ชฤ?ฬ?ำ?wู?ก??ใ?๚ๆ?T๊?<ํ?ภ๏?ํ๑?ะ๓?r๕?ภFฟฐH-ฟ%?R?dแพะV?>K๒Vฟ๒A?#V??e>4Nฝ?>%,+?}ร|?tญ@=Fฟงใ?ไ>ด<ฟ^@f?อkฟ'คW?๙?ฟจn='"?โฟvn้>o>,?ฎ_ฟ7อพม๘h?{๚?A๔6ฟ7cฟx"=eญj?mS?ดq=๔?ฟฟ๋8*ฟ?รธผ?ปmr?;x?ฒเ7?l์><Cพ}ฟh@SฟU-yฟf~ฟลhฟ?ฟใ8	ฟ`พ๎ฮฝ\(>;ต>ฬส?]+?ผH?m์^?ฟn?ค?x?y~?0??ฟG~?K?y?กธs?^๒k?c?Y?~rO?R+E?ื:?)0?&?ฯ?าb?๊V
?ฒ?๏๒>Sใ>Gิ>ฦ>น>''ญ>Aก>?ฑ>ฆ>V?>H&t>~c>๐S>gjE>F?7>R9+>ีo>ูr>"5
>2ช >|๏=d??=Fฯ=1ม=	ัณ=u\ง=ัร=!๘=b๋=!{=ถi=บY=2jJ=2_<=HM/=@##=ั=แG=ถy=Kณ๔<Cทใ<เ่ำ<>3ล<็ท<พลช<็๊<ฑโ<๓เฝ?o<?S๛?็uฟs??ฟ'ฟI??y?๘ฌ?S?}?[>?O"พsทฟdด==i[?๙uฟk5-?ภษ฿พ็Sว>๗	ฟฒuO?ฉฟ=ภE?hm๕<_ุcฟ|c=?$#๛>รjฟe:ิพC$Z?K3??๋พ๎JฟมฬพIW?ย??5U*?ัีฏฝ'5?ฟT๏ฟ$YNฟะ}คพึRt>w2?ืs?1}?๘๑V?A?Yึj>Jไฝึพพm*ฟฉXฟrํsฟ๋dฟ๗|ฟ=toฟศฃYฟล>ฟBเฟ
น๛พtศธพX๚mพr4฿ฝ:ฃ๓;ฟ?์=s\>6ช>ฅฦ>([์>๗?g?ขG#?ภ /?ุS9?'fB?ZJ?QQ?gW?ท\?๚Xa?be?gๅh?ฅ๓k?จn?~๊p?๋r?ฉt?ฉ+v?{w?๗x?<y?๕tz?2{?่ึ{?Te|?ฟเ|?ฐK}?Wจ}?๘}?&>~?bz~?ฎ~?ย?~?็?ฯ$?,B?[?ขq?ต?;?ฃ?๎ฏ?ฉบ?๔ร? ฬ?๘า?ู?;??ยโ?ญๆ?๊?ํ?๏?ย๑?ซ๓?R๕?ชฟj>>๏"w?ะdฟF^`?ฏ~ฟเตU>์?๒x5?ๆ9้>๎?Y<m??V?ชTญพH#fฟ{E?
ฝO*ฟi??Kฟ๏11?ฒแษพy
พี,E??~zฟ->FAG??JฟmหฟพZ?8?ฑ"ฟ?mฟP7ผ๓`?S\^?8eฺ=qฤ2ฟ๑ฟu5ฟ'2กฝZ?B#n?>{?9??ข๏ฎ>p_ไฝgชฟ๙NฟWwฟdฟu6kฟฺCฟ ๕ฟ(\ฅพ๎าณฝั#>*"ญ>ยl?ฆร(?'F?5X]??ฉm?KVx?๏.~?ใ??๚~?้nz?@Mt?Uขl?Rฺc?๔NZ?IP?]F?.ฑ;?ฐq1?๛b'???&??d?฿B๔>๖ไ>๗พี>ึฝว>}บ>\$ฎ>ฺ}ข>ฐ>,T>lฟ>
u>Kำd>U.U>hF>_๒8>ภ;,>ษ` >qS>p>5m>ำ๗๐=ัNเ=uสะ=Wย=Tโด=้Zจ=ถฐ=ฉิ=ฌธ=3|=Tk=ุหZ=bK=	~==>X0=ถ$=Sธ==?A=(๖<*ๅ<ท+ี<ฎ_ฦ<~ธ<๎ษซ<	?<ฤ<?jL?๕{?[พฤ่พ๚๖>1?=?\zฟ๘Gวผช4?ิๅc?V๎P?shภ>7{ฟPโpฟNAเ> ็"?็ึฟ.p\???)ฟใ?ตร8ฟcBk?ญ<}ฟฮE#?<@S>|๕sฟWบ ?!$?D?Zฟ_"ฟf@I?@งE?DRฝพ๔ฟ}f๔พ?ด?>N~?_>7?ฦ8ฌผ4ฟฏ4ฟฆ?Uฟ๊ๅปพdG>z4*??p?Eg~?[?ข4?้>ขฤฌฝEสพ๏ู%ฟ	Uฟ็Grฟโฟ/?}ฟd๋pฟฐ[ฟ์~@ฟๅ!ฟ4 ฟฺHพพชชxพ'๓ฝํ(๕บแ\?=]bT>V>QYร>ฺf้>฿ด?(ึ?ํ>"?K.?{8?ๅฑA?ผI??ฦP?W๎V?ลM\?9?`?5e?เh?/ทk?gn?ฺผp??รr?ญt?ฯv?)aw?x?ฤy?dz?๎#{?9ส{?WZ|?9ื|?pC}?1ก}?k๒}?ศ8~?ผu~?ช~?Eุ~?โ?~?1"?่??ฅY?๏o?=?๕?oข?๙ฎ?ีน?<ร?aห?nา?ุ?ิ??hโ?`ๆ?ฯ้?ศ์?\๏?๑?๓?1๕?
|ฑ>ฒ(f?ษฺ>ธ?qฟล2x?T_.ฟฎูพํO?ฬy?จๅV?)d?๚h?\ร	?2,ฟบ๋.ฟy|p?ฬZฐพVrพO
?	6ฟยิ ?ิ?;พqธฐพี`?e3mฟ{^>แ]?ผ}2ฟท ฟAฤG?[4?Nฟ%vฟEiไฝ!U?ฑูg?ซ+9>*%ฟาภฟ@ฟmุ	พด ??Hi?D}?z:F?ใญม>t๊ฝษB๘พชฐIฟึJuฟHฟถmฟใFฟซฟhFฏพฦฺฝ(๒=๕"ฅ>พ ??%?/mD?Cป[?l?งw?ฟ?}?่๛?Rม~?]ฺz?[?t?YOm?d?ว[?Q??F?ฃ<?JG2?ณ3(?nd?+้??ฬ?ฑ?๕>๊ีๅ>M๎ึ>ใ?ศ>ปป>a!ฏ>Mkฃ>Co>$>m>ซ?v>๚'f>xlV>YผG>j:>#>->ณQ!>4>ทื>30>!c๒=8กแ=า=|ร=๓ต=ZYฉ==0ฑ=๕=`~=~l=๕\=าL=?>=3c1=*%==F๖=B
=า๗<mๆ<nึ<ว<ฒน<ฮฌ<*ฯ?<Wฅ<m p?+เ>gฟ0หง>฿zพDn;?๑ตgฟYฟs`>?ำึๆ>ร๔ฝ~ลWฟ@n=ฟD๋:?ยฏ>ญUpฟHทx?lWฟ
J?y8]ฟ"ฆ{?zDpฟUณ๕>ภ>Bh}ฟeC ?@7?ใCGฟo ฟCฎ5?u๖U?6โพ"g~ฟซ2ฟู>ว{?1CC?หฯ3=:J)ฟ~ซ}ฟ-]ฟ๙?าพUQ>?!?๚l?ฉF?[ไ_?Oฉ?a>งiฝ๋พพ@.!ฟใูQฟ4pฟ~ฟD4~ฟงQrฟ๓ฏ]ฟ๔ไBฟ-,$ฟาZฟมยรพัฉพะ๏พ*7ผอธษ=NL>BV>ค
ภ>pๆ>b?ตฉ?5!?ศ,-?%ธ7?ภ?@?ุI?<P?
uV?๎ใ[?๗?`??มd?๗Yh?czk?E2n?๒p? r?dt?ส๏u?Gw?์px?.sy?Sz?6{?wฝ{?IO|?ฅอ|?#;}? }?0์}?b3~?q~?|ฆ~?ริ~?ุ?~??ก=?ซW?9n?ย?ฌ?Sก?ฎ? น?ย?มส?ไั?ุ?l??โ?ๆ?้?์?)๏?k๑?_๓?๕??z?ใ`c?!ำพช๎ีพเ
?0ะRฝ๑ฬbฟธฉ>#1q?๚ฉ??๓?ุ^?a> gฟEมพจ??ฟสa=ำห>,hสพ9>F=Iฟืs?\mXฟทผผ+n?4tฟ:ศ9ฟ@j2?H?์้พฏ๋{ฟ\EPพH?ึีo?$>ฟ |~ฟ2่Iฟ6จBพว๊>ฆ฿c??~?แL?อิ>?ฝc็่พบDฟอsฟใฟpฟขงJฟ๎7ฟปนพmา พฮ๔อ=->็1๙>ม-#?G8B?งZ?(gk?๑v?}?A๒?ฦ๖~?ฅA{?๏ku?g๙m?gYe?่[?ฟ๒Q?IตG?๓`=?๓3?จ)?(.?ซ?:?ศ?ม็๖>m็>Gุ>ฅ๙ษ>พฆผ>5ฐ>Xค>ถM>๋๔>ZE>+lx>|g>ชW>8ๅH>f;>{@.>B">>๙จ>-๓>hฮ๓=๓โ=ล@ำ=ขฤ=แท=ศWช=y=ด==S==ฬแm=b]=ฟN=ณป?=(n2=&=ษ=yอ=า=๙<๙ว็<bฑื<ธศ<ซษบ<Mาญ<Jมก<ช<D\>ญ>๋พก8iฟ๊h?งWฟ$ฉ?BwํพญqฟซพฃหQ=ำผก฿?พดA}ฟ[ฺพgm?ฝ%<??HฟZ?fฬtฟ%k?uKuฟqี?3Yฟ้>ป?๎ฟซน>KO?ท 0ฟ27ฟฐ?ฟอc?6พ)ฆzฟ3#ฟฎณ>w?ฐTN?๖|?=,bฟ U{ฟวcฟR้พbึ=ำ}?ฯ?h?้ฮ?๋๚c??๕#?^ๆ?>>2๓ผ๙?ฑพ]kฟNฟฎnฟV๗}ฟฤณ~ฟ๎ฆsฟ่ก_ฟพ@Eฟ๛ฦ&ฟAฟ๙5ษพx๚พซพฦงผฺธ=k6D>ฉ>นผ>ตvใ>๑?|?* ?8A,?ุ่6?ปF@?T~H?rฐO?๛U?y[?7D`?qd?ชh?@=k?"?m?ษ`p?
tr?HAt?ัu?์,w?4Zx?z_y?Bz?h{?ขฐ{?*D|?ฤ|?ษ2}?ฤ}?๋ๅ}?๕-~?\l~?jข~?<ั~?ส๙~?๊?V;?ฏU?l?E?b?5??ญ?)ธ?สม? ส?Yั?ื???ดแ?รๅ?H้?S์?๖๎??๑?9๓?๐๔?ฏj5?ฅ&>vฟ/mป>่อ3พE.?&o|ฟยฬพ]M?ค d?ๅ_?ำ?Vพ๕ณฟขjณผห5r?ถ0Uฟ~Iฏ>e?ผ<0ฃพใLY=ฌ?>ป๐0ฟด5~??ำ<ฟ>็?พึz?+ช๎พูOฟอQ?ฎ'Y?Xaทพุ4ฟ[Dพ5:?}Cv?Pษฆ>+ศฟอ$|ฟSฟtฺzพ<จา>ฝ์]?i?+S?>ๆ>jป$GูพvR?ฟไpฟำ?ฟJ>rฟNNฟXฝฟแยพม5พ๊ฐฉ=b>2G๒>?U ?๚??rgX?ป9j?J4v?แ'}?ํโ?U'?ภค{?๛๕u?}?n?;f?Eฒ\?ฦฤR?%H?7>?ซ๏3?ืา)?B๗?{m?+A?
z?a9๘>V่>ๅKู>ห>ณฝ>ูฑ>ฝEฅ>
,>%ล>1>ุy>ัh>|่X>J>U1<>วB/>l3#>๕>4z>"ถ>จ9๕=๔Eไ=ๅ{ิ=วล="ธ=4Vซ=Ww=8j= =?=Eo=*ญ^=๋:O=ฺ@=y3='=n=ชค=อ=W๚<฿"้<8๔ุ<?ไษ<Aแป<|ึฎ<kณข<?g<?4ฟ|ฟถcพ
;n?ส|ฟ๘\L?Q*>ื๗vฟ0Iฟ๚ะ่พ@@๘พวSฟ^vฟ๘BEฝH๐?%ฦฅพนฟคp?๎ฟk}?ถฃฟZw?'9ฟ4ว๑=ี?,?เv{ฟ?D\>Y|b?๎?ฟ%BLฟ`?>o?	"กฝ๎นtฟlโ/ฟ>?>L	r?eX?๖1>็ฟ3xฟ๎Siฟ:?พ6t=ฤท?ซงd?ุ??ฮg?๙*?ฐ!ฏ>4ปvฅพ๖ฟ|-Kฟปlฟ์M}ฟงฟ!๋tฟJaฟ)Gฟ2Z)ฟhฤฟSขฮพ!Gพไ7พฺ๛๓ผ_eฆ=|<>ด๙>$fน>๐zเ>"บ?;M??T+?6?ึ???G?,$O?U?[?๗ๆ_??d?๚ฬg?ฦ?j?ณวm?\2p?บKr?It?:ณu?w?ZCx?ชKy?ื0z?๗z?บฃ{?๛8|?Qบ|?d*}?~}?฿}?(~?กg~?Q~?ฏอ~?ท๖~?@?9?ฐS?ฦj?ล~???ฌ?Qท?ม?~ษ?ฬะ? ื???Yแ?uๅ?้?์?ร๎?๑?๓?ฮ๔?ท?Wพ^ห1ฟCUฟ}ยl?ฉLฟ?ty?KN/ฟcษCฟ$Dี=๕P?ทศ?ศ[>N$ฟ!๛pฟฦRฌ>@ปH?ฑvฟ่ฃ??~พำ=hj;พd>๋>ผ๒RฟR?มEฟฏ|ฑพฦ~?iซพขaฟณ?>[Tg?+พ8๑ฟ%รพ*?๘{?~ส>ธ๒พ_ฝxฟฟh[ฟึ พผ๕น>"sW?C๙?GY?ฅ๘>=ซfษพ9ู9ฟจเmฟe๋ฟ0FtฟฝืQฟX/ ฟbฬพ9'พJ_=%โ>ศM๋>u?	ด=?ถฐV?Oi?ณou?9ร|?ํอ??R?ฌ|?}|v?Do?ฮf?์y]?S?_I???qย4?Aก*?บฟ ??.?ฒ๚?5+?s๙>้>&zฺ>J4ฬ>ภพ>Lฒ>น2ฆ>>
>D>๔ส>ฦD{>d%j>]&Z>ภ6K>5F=>E0>;$$>ี>jK>y>฿ค๖=Hๅ= ที=๗์ฦ=`'น=Tฌ=3d?=นF=ศํ=๏K=<ฉp=C๘_=oP=Z๙A=4=?'==U=?{=c=๛๛<ล}๊<7ฺ<kห<ื๘ผ<ซฺฏ<ฅฃ<OI<5?zฟ08ฟ8?9ฝย>ศฟjf>ึ:?a๏$ฟณ~ฟลVฟฟwVฟ%_}ฟูPDฟtสฌ>q?Nแฟ!ฬพลL?%?wฟฃ~?ญ{ฟ๖`c?y
ฟnใฝ๐K?N pฟ-=อ4q?ื฿๑พๅษ]ฟตK?>ิxw??ญ<ฏlฟ4P?ฟฺG>ฝk?หga?]$r>xไฟQHtฟnฟD
ฟf๘j<ีฏ?g๖_?dู?ฟ\k?ศ0?06ฝ>พhอ<m.พฝขฟฑGฟ!ฎjฟี|ฟโtฟ(vฟ๛\cฟูIฟดๅ+ฟ*qฟิพฅพ6V"พ? ฝฑถ=m?3>UH>gถ>พ|?>d ?9?ห?๗f*?]G5?ุ>?๎<G?2N?WU?	ฃZ?7_?6ฮc?่g?๕มj?๗m?ฎp?0#r?๛s?ฐu?๘v?`,x?ฝ7y?z?่z?ฟ{?ป-|?ฐ|?๑!}?,}?Gู}?#~?฿b~?2~?ส~?๓~??ถ6?ฎQ?	i?C}?ศ?๔?ซ?xถ?Sภ??ศ?@ะ?ฆึ?1???เ?%ๅ?ฟ่??๋?๎?ๆ๐?์๒?ญ๔?๙ปoฟํ๋}ฟo๒ฝi3j?ค~ฟd?ึฝำ~ฟOdโพyU=ิfน=พาCfฟ=ฟ?&?}K?fปฟmN?{9 ฟ?oณ>QUัพ-$?slฟู#w?่ฃ้พU,?พฝ๏~?์ภIพตoฟพิฦ>2lr?ๅkพํ~ฟพ=๎พy๋?AK~?+>ํ>kฒาพItฟZ๙bฟXดพๅภ?>ยvP?เ?<^?ทฑ??๗=ฉJนพi04ฟฑ?jฟึฅฟ{%vฟผBUฟ`$ฟ'ึพ	ั:พปB=ธ>Fไ>๓?้d;?๑T?๏ฦg??ฃt?X|?Bณ?ภy?g^|?r?v?ผๅo?og??^?ดcT?ฆ2J?๘฿??C5?ใn+?!?ฝ๏?อณ???๘ฺ๚>Kี๊>จ?>,Qอ>Yฬฟ>ณ>ง>Q่>Je>ข>เฐ|>งyk>(d[>j_L>[>>>G1>%>?ต>>?;>๘=๊ๆ=๒ึ=eศ=8บ=Sญ=Qก=9#=ป==๒r=ZCa=AฃQ=,C=5=๗๕(=v<=S=V+=ฺo?<ซุ๋<แy?<ู=ฬ<mพ<ฺ?ฐ<ฌค<ข*<ณพa8>33~?๙พฮพmจื=pm่พ~??รฝ=eฟ๏ฆฟ็๒~ฟZัsฟป฿พ;?+?^fB?ฎณXฟ๐];=ฏg?]ฟwรo?$?iฟxD?๕ลพพญสc?4:^ฟu|บฝe๛z?Wบณพo่kฟ?ค>}?Gฑ๗=แbฟ๒NMฟl์=เ?b?ขQi?:>Lอ์พีoฟ sฟโฟฒ?ผIิ?>?์Z?[?ำฅn?ุ5?พ ห>OศV=ํฎพiฟษDฟะhฟ"ด{ฟoถฟ๎?wฟ฿%eฟgLฟbi.ฟmฟฎeูพ?ำพap,พ?*Fฝ)=`?+>๙>eธฒ>)|ฺ>ศ?>์?y?Hx)?1u4?o>?F?	N?T?๓6Z?๚*_?|c?t>g?ฯj?๐[m?ฝิo?l๚q?ตืs?๚uu?q?v?Dx?ณ#y?1z?{ูz?ฑ{?k"|?ฤฆ|?s}?ะ|}?็า}?{~?^~?~?ฦ~?๐~?แ?a4?ฉO?Ig?ฟ{?x?ั?ช?ต?ฟ?8ศ?ฒฯ?+ึ?ฦ??กเ?ึไ?z่?ก๋?\๎?น๐?ล๒?๔?(Mฟ;(๛พฐ?-?r8ฏ>๊'ฟ`6ศ>8?ฆ_ฟ0Zฟh่พงคลพๆ0ฟqแฟXเฺพทa?a>ํhoฟํwq??]7ฟุB??ฟBzK?fP{ฟม๕e??]พ๕!ฟXx??@aฝmyฟ็>]Iz?ี3ฝ?ยyฟIฉฟ?ุ?ฝH?ฌฅฑพฮnฟืตiฟ<?ฮพd>?๛H?aA??ฝc?O.?~t?=โ๗จพyY.ฟฃๆgฟ5/ฟ??wฟXฟไึ(ฟถจ฿พaNพ]z๒<7	y>0?>;?A9?๊)S?ฅf?ศะs?ๆ{?๋??๐ด|?ุ~w?เp?ส9h?_?0U?GK?จฒ@?"e6?พ;,?รN"?ฐ?|l??๏*?>์>ี?>ยmฮ>iุภ>ด>;จ>Eฦ>55>;P>ู~>ฮอl>?ก\>M>หo?>gI2>ฝ&>h>รํ>ไ?>6{๙=?<่=&-ุ=ฮ7ษ=าIป=hQฎ=ๅ=ข=ท?=N=ส=ฆps=ob=jืR=?6D=๔6=i๎)=ฏ#==*=๓=ไ?<3ํ<ถผ?<Gjอ<(ฟ<	ใฑ<อฅ<๔<C6??_??ว;?ฟpฟe<A?kฟM?Z?[H๙>:็ฟhdฟH(lฟุ9ฟ#๚ผฦmg?O2๔>=ฝyฟQต>?ช>ค2ฟgQ?ฟJฟ?X?#Cพะ๛เพ2?t?รAFฟxพำ?นวdพ*fvฟ๖!W>ฮภ?฿ด`>8VฟหรYฟ็	=UY?บp?{ธ>X๔ะพp%jฟํvฟฟฝาี้>U?ฉ~?]จq?ถs;?F?ุ>Yฃ=
พฑฟn@ฟdHfฟ็รzฟIใฟ_Pxฟูเfฟ?FNฟๅ0ฟบฟQผ?พพ"6พ<:lฝ9ขb=wธ#>ญ฿>)^ฏ>7yื>๖h๛>ธน?๓?(?ข3?๎e=?c๘E?{M?T?NสY?=ฬ^?)c?๖f?QEj?%m?ฅo?nัq?ดs?Wu?ฅยv??w?y?ท?y?Uสz?|{?
|?่|?่}?iu}?}ฬ}?ํ~?FY~?โ~?๊ย~?cํ~?,?2?กM?e?9z?&?ฌ?ฉ?รด?ุพ?ว?#ฯ?ฐี?\??Eเ?ไ?4่?e๋?(๎?๐?๒?i๔?ๆ=ใลฯ>ใฤ?Jธแพ==๓aพxn?u`โพ[๛ฟฅVฟ๓ืEฟpฟ๗ฉlฟlุIฝฺล}??๑ฝwXGฟรฟ?Lุ`ฟัJC?(ฯGฟ?h?ิ๙ฟlhL?I๑ฝ?g@ฟี฿k?ด=๋~ฟ9ญ>ั~?า=y็rฟP?ฟีๆ>ถป?บ/?ปพิPhฟ่oฟn?่พb.Z>๗A?~?ืwh??s?ำ>)sพ่U(ฟ-dฟ~ฟiyฟ(ผ[ฟ]-ฟญ้พw&aพBซA<๊h>ึ>ฃ?,ญ6??YQ?}4e?๖r?wn{?๊l?ธ?F}?ฏ๚w?q?ฃ๋h?pฤ_?ม๛U?{ิK?+A?57?ะ-?R#?ํo?พ$?ง<?Vz?>JRํ>ป?>ฯ>;ไม>|ต>ฟ๘จ>ค>>ฟ>ฐ>ู!n>|฿]>ฐN>@>K3>o๖&>ฬv>ๆพ>ลม>Vๆ๚=้=2hู=3]ส=[ผ=ษOฏ=ป*ฃ=3?=U=#=Wิt=ูc=T=อUE=ไค7=ูๆ*=็
=m=฿ป	=ฎ, =u๎<??<ตฮ<?ภ<7็ฒ<ํ{ฆ<Gํ<ร[?7๙i?1๏-ฝวeฟู??tฟwผ>0e?
C<_ฟLv"ฟ}ืฑพQ4ร>aฐ?cว>ญ6~ฟ ?HP5=ธส๔พ+%?ำ	 ฟxึ>อ?`<ฟง6~?ค?(ฟ?ฦพห ?ผฝศ}ฟ,ฑฤ=พ{?*ฉก>GคHฟ?dฟณ๐:ฝN?กดu?สื>Xดพ๕cฟ!zฟ๚'(ฟRo๘ฝ์lึ>?ุO?ีZ}?xct?๋?@?บkๆ>/?=ธืfพSYฟPจ<ฟ๐cฟ;ผyฟl๛ฟhOyฟะhฟปmPฟฯX3ฟVฟZไพฟNกพ7@พ"ฝว5?=ึ>?P~>พฌ>๒sิ>หต๘>=?โ?ี'?ฮ2?ซ<?๒TE?์L?๙S?]Y?m^?ึb?cฎf?}j??๎l?vo?6จq?Vs?
8u?ดงv?งๆw?G๛x?%๋y?ปz?\o{?|??|?P}?๘m}?
ฦ}?W~?oT~?ฑ~?Iฟ~?>๊~?s?ฌ/?K?ยc?ฑx?า??จ?ๆณ?พ?๎ฦ?ฮ?4ี?๐ฺ?่฿?5ไ?๎็?(๋?๓ํ?_๐?w๒?G๔?นฉ`?qฒy?o?์sฟ้ี4?VฟAv?ัI>*๐Vฟึฃฟ?zฟ?ย~ฟT๛/ฟ/@ฌ>y?Teโพ%ฟล
x?yฟภตf?VะgฟsVz?แyฟj+?t=ฆYฟuาY?xk>ใวฟIyู<ฯ๓?*A.>iฟ!๓0ฟUพ>}๖}?,=&?ง<Zพ์ุ`ฟtฟฟ%>๋8?q|? ศl?G}?ูน7>_มพ@'"ฟ
aฟฏ}ฟฝฬzฟฆษ^ฟC*1ฟza๒พ2tพฤ[รปPX>?ฮ>
ฃ?ฦD4?ำO?฿c?	r??๏z??A?ะ?gU}?๔rx?'ทq?๙i?ฤ`?/ลV?AฃL?TB??8?ำ-?<?#?</???r์?.ษ?>๎>/฿>
ฆะ>ฯ๏ย>(ถ>ๅฉ>ส>ฝิ>.ี>2z>ษuo>_>?ุO>&A>M4>็'>(W>>ก>nQ?=Yแ๊=8ฃฺ=ห=7lฝ='Nฐ=ค=ญธ=ฮ"=2H=8v=$e=น?U=tF=ีฏ8=J฿+=๒=ุ="
=็ =Z้๏<]B฿<#รฯ<,Wม<f๋ณ<nง<ฮ<u๕>๖ำa>+?Iฟฐhพ๑35???ฟ<ยพสC}?G?a&Yฝ??Kพ๘;ษ=ฅ๋9?฿q?
lพ?eฟฅV?~U}พภdพไ?>=ูพฺ-V>^>ื#>ฟ?a?ฟRฟ*y?Zv)=็่ฟ5ผ"D|?dZั>??8ฟ]ณmฟ,ํ พIฎB?Yz?kศ๔>Wพ']ฟ?ก|ฟXZ1ฟ"*พlขย>ำI?ุ{?Uึv?มF?ฦ๓>ทm	>?XMพ4?พห8ฟ:aฟ7xฟึ?ฟ๖<zฟจ,jฟRฟUฤ5ฟ1์ฟR้พฆพ^ฃJพี$ฝฉล=h>๖?v>-ฃจ>blั>L ๖>Q?๎ะ?ฆ&? ๙1?X๐;?นฐD?>\L?CS?Y๏X?I^?6b?ศef?Sวi?ธl?]Fo?ล~q?[ls?ฯu?v?'ฯw?ๆๆx?yูy?ศซz?b{? |?|?ฌ?|?{f}?ฟ}?ธ~?O~?y~?ขป~?็~?ต?M-?I?๚a?&w?}?^?ง?ณ?Yฝ?Gฦ?ฮ?ทิ?ฺ?฿?ไใ?จ็?๋๊?ฟํ?1๐?P๒?%๔?วจ`?opB??Oพb๙`ฟNิ?Y!~ฟ0น??.?eืพ 6dฟ!uฟ?Zฟฅ๎จพะล+?[S?FB7ฟ๓ฏพ[?ภฐฟ:{?๔ืzฟภ??:Iiฟ?-?	?p>"ึlฟ[B?ฟyป>|ฟ?sวฝรญ}?ใผ>๖ ^ฟ๓_AฟE>Mz?i[4??๒พ`nXฟทxฟXWฟq๏เ=าย/?Az?aฌp?<I%?แa\>้ฮmพฯฟ p]ฟ๐ฅ|ฟร|ฟทaฟ35ฟ๛พ?พยผpG>กว>แ?,ิ1?yกM?ฟb?j,q?kz?์?ธใ?R}?ฅ็x?GLr?ษGj??@a?แW?pM?ซ#C?๘ั8?.??$?๎?๛?ไ?ป ?vอ๏>๑[เ>ปมั>$๛ร>ขท>Oัช>[_>Yค>>๛/>ษp>vZ`>^Q>พญB>O5>ธื(>}7>a>yG>~ผ?=3์=9??=๐งฬ=d}พ=Lฑ=`ฅ=&=๐=@=ตw=งof=?sV=jG=ฤบ9=บื,=Vู =ฬฏ=eL=mก=>D๑<1เ<๏ะ<มnย<๏ด<-`จ<๋ฏ<?x๕พ&ฟ;ถzฟQ๔>๎=๖๗=฿๓Kฟิ4;?v>h???็>ฃฆ>ุ&?๚ฉq?้ะ=??2ฟrภ2ฟ8cw?ุฟณ_I=าD>k}LพG'#ผกืา>{<[ฟ+ัx?]ยพ?P&ฟ8n?h2>xศ~ฟ	พ|#v?๔?>ม'ฟึuฟ๖GRพ฿ญ5?dP}?rน?มrพ?qUฟs~ฟ!:ฟ๘ณWพTฎ>}C?> z?< y?โK?9u ?K)%>ธ3พ:๑พศึ4ฟ ๖^ฟ๕fwฟํฟ๚{ฟIฝkฟ<Tฟ'8ฟm|ฟโ๎พถซพTชTพต#ฏฝฅ๐<๔<>bio>Bฅ>bฮ>H๓>แ
?@พ?Qณ%?#1?D4;?บD?มหK?๎R?X?ญ]?`/b?สf?ำi?฿l?co?Uq?-Hs?i๙t?^qv?ทw?hาx?ตวy?az?ปT{?๔{??~|??๖|?๔^}?น}?~?ฌJ~?<~?๖ท~?ๆใ~?๔	?๊*?vG?0`?u?%?5?ฆ?)ฒ?ผ??ล?sอ?9ิ?ฺ?,฿?ใ?b็?ฎ๊?ํ?๐?(๒?๔?ษึ= ฝฬ`ฟmVพA?3แ.ฟBษนz?Gู>มณฟ.7ฟฤK	ฟ"สฎ=XNg??>\hฟบฌe=)&+?pยrฟn?Tีฟภ8y?JใNฟ"Dฐ>โฮ>yฟ"อ&?`?>;๒sฟbaพชx?นภ>+Pฟe Pฟ็จR>ิu?"vA?๔?ฝBOฟพ์{ฟEฟ1Nl=~&?_w?7#t?	ิ,?W_>ษิKพ	OฟเYฟDl{ฟใ}ฟ?dฟM%9ฟฬXฟ)พ7*ฝุส6>_Xภ>9?y[/?นK?@a?ฉ<p?ษ฿y?๑ุ~?๏๑?ๅ}?มXy?a?r?๒j? ?a?ำRX?<N?ฅ๑C?๙9?Ig/? e%?Eฌ?๕J??J	?ฒ ?\
๑>?แ>?า>;ล>้?ท>Zฝซ>ห<?>?s>สY>ฒๅ>Ur>ัa>ฎ)R>GยC>Q6>Nศ)>ส>*2>K
	>'?=ปํ=5?=Hออ=ฟ=?Jฒ=/๑ฅ=q=Hฝ=Lฦ=a?x=ถบg=จW=7ฒH=ณล:=)ะ-=ภ!=๛=จ=ฬ[=!๒<ศแ<?า<Vร<ย๓ต<LRฉ<=<็[ฟhฟี๖พๅv?'ฟ๕:???ฟZ\[>มๆ}?gV?ี2?ซX?โ?e?>|:Rฟsไึพไ?๊`>ฟข>Ii{ฝTษ=(jพฌฦ?ึYpฟฬ=j?์]cพ?2Bฟrd^? B>๊ผyฟร}พไ+m?ฺ ?มฟ6zฟฎ"พ	จ'?ฤF?!{?/?5พา)MฟูฟัvBฟjพะ> ?<?ขาw?เz?๓O?๋๊?+ล@>_๙พหๆพ๔ห0ฟตT\ฟvฟวฟdใ{ฟ?mฟวVฟr:ฟงฟษ๓พ๏โฐพูซ^พดยฝFบฉ<๘>\๐g>ศ฿ก>Vห>p๐>ๅ?ช?ฟ$?0L0?Vw:?๕eC?:K?๙R?*X?\L]??a?kำe??Gi?^Il?'ๆn?4+q?อ#s?ืูt?๙Uv?รw?ฬฝx?ืตy?ใz?NG{?โ่{?่t|??๎|?bW}?yฒ}?c~?ภE~?๘~?Dด~?ณเ~?/?(?bE?c^?
t?ห?	?ฅ?Hฑ?ีป?๘ฤ?แฬ?ปำ?ชู?อ??@ใ?็?q๊?Uํ?ี๏? ๒?เ๓?WMฟ{7Wฟฦ(oฟ@?>ส?=่ๅ]ฝCแฟ๎l?	#?ฏำ\ฝgพHผๆฝฝซ๘>ธฌ?u>์โ~ฟFBบ>ศ%ู>ฺผSฟNs?๙vฟs^f?พห+ฟ๏#>x?0ฟO
?B?ugฟIญพหo??|๒>ณ@@ฟษ]ฟฯf๖=u?n?zzM?รฑปฆๆDฟk6~ฟEg$ฟฟฐ;7ิ?Yt?์*w?ท4?์a>h)พลจฟUฟTzฟ๑?}ฟ0gฟr =ฟ๗ืฟIcพใสrฝM&>
น>;t?สฺ,?ศI?ฒ_?ฯEo?Ny?P~?<๛?&~?Fฦy?tms?ะk?#ตb?Y?๔O?nพD?k:?/00?)&??i??บ๙	?'Y?หF๒>ฉณโ>4๘ำ>ฦ>?๘ธ>;ฉฌ>ก>BC>๘>X>๐ps>ีb>๋QS>มึD>S7>?ธ*>๘>5>อ	>BI >โื๎=,T?=๒ฮ=ตภ=2Iณ=๛?ฆ=N==X=cz=ฤi=&?X=ัI=?ะ;=ศ.=ยง"=)^=๋?=+=๚๓<ึ
ใ<jHำ<๊ฤ<๐๗ถ<lDช<r<ฎ4ฟฯ?
ฟsขถ>๎ส[?~ฟร?rีKฟyๆมพ\E?ฐ??Vรs?พ^~?;ว_?ะmN=ตxฟฒะพฝun?่ีgฟศโ?ร;พX(>ศM฿พห=?ท|ฟ"GT?)oฝe|Yฟ4J?ฟฺ>ฺpฟ?ทพีwa?
)?ญฟ๒#~ฟ<0ธพ๕ฐ?
??l#?๑ฝผ;Dฟ?ฟVJฟตพ,T>๎5?mPu?ซv|?๋T?ดB?
<\>! พผไ?พซ,ฟ?Yฟ.ตtฟฤฟ%|ฟณnฟ
Xฟาิ<ฟฤฟื๗๘พ
ถพชงhพdีฝ]E<ฝ๕=?s`>	{>BHศ>าํ>ญ?ฐ?วส#?dt/?น9?kฟB?ฒจJ?fQ?ฟขW?*๋\?fa?ฉe?ัi?l?ชตn?q?;?r?บt?n:v??w?ฉx?฿ฃy?P}z?ฮ9{?.?{?ลj|?vๅ|?ลO}?แซ}?ญ๛}?อ@~?ฏ|~?ฐ~?{?~?e?&?KC?\?yr?p???ค?gฐ?ป?Nฤ?Nฬ?<ำ?<ู?n??๎โ?ำๆ?3๊?ํ?ง๏?ุ๑?พ๓?GปoฟYฝoฟ๖ษชพMty?(^ฟX??ฦvฟ฿โ?*s?ํ็>บำ3>O?ช>	M?,q?j๛ฝศBxฟ+ซ"?จ>ั๏$ฟฎ`W?`_ฟ๋WH?|ฟฒ๛ผฑa2?ศ~ฟก*ศ>ฆ~8?ฏ?Vฟ?ฬ็พฑํb?%บ?j.ฟmhฟ*F=7ฌf?(WX?qน=?9ฟฟ7,/ฟ้-@ฝ2ห?lฃp?!ยy?g;?2ค>ใ5พ๛?ฟุ?Qฟfhxฟวธ~ฟศบiฟฤ@ฟHฟZถพ๓บฝZ>zคฑ>V?=R*?ะG?B>^?โGn??ตx?
Z~???รc~?20z?๙s??l?lc?uูY?๕ฯO?E?6;?H๘0?i์&?-'?ท?จ
?p??ร๓>๕?ใ>๛ี>ญว>฿๓น>๔ญ>H๗ก>>?>์P>oฤt>Cd>zT>,๋E>jU8>]ฉ+>Mุ>9ิ>แ
>พ? >*๐=฿=์ะ=ืฐม=Gด=ฦสง=*=ฝW=bD=ดฦ{=ัPj=GZ=ฮ๏J=?<=ม/=๘#=W5=-ฅ=ะ=็T๕<จMไ<ืtิ<~ตล<?ท<6ซ<เS<ถณ>ชณ>fVq?f>U๙LฟพวL?OaพฑfVฟข?>ฬPd?|?ฬRq?bH??ตซพN~ฟฆแy>|ฌE?d\}ฟษC?ฤ`
ฟ๋๚>}^ฟฐึ\?แฟ7?พึฺ=?kฟีy1?_-?Cdฟ?๙์พแ)S?หc;?Q๓ืพฺฟย?พ,??Mx?)	0?จlฝีฎ:ฟำทฟrปQฟนฐฎพณฝ`>Mน.?|zr?.ย}?h๘X?.{?ฃw>ษfฬฝ9?ะพฑt(ฟหVฟ๊9sฟX=ฟ1C}ฟ๓pฟ้Zฟ?ฟฌ ฟ(?พึ+ปพrพ\่ฝแ^;6Yๅ=f๔X>P>ฺ7ล>๋>๚s?ั?ี"?ซ.?๎๚8?B? J?4Q?ฦ2W?z\?B1a??e?Oวh?yูk?๊n?ฝึp?vฺr?/t?ฝv?ูow??x?ฯy?งmz?;,{?jั{?`|???|?H}??ฅ}?๎๕}?ำ;~?_x~?าฌ~??ฺ~??ญ#?1A?ยZ?ๆp??ฎ?ฃ?ฏ?Mบ?คร?ปห?ผา?อุ???โ?ๆ?๕้?้์?x๏?ฏ๑?๓??ึWพผlพ ?l>V?O๎{ฟLy?๎๓mฟhึผex?;eV?๑ฉ?โv6?ฟะy?&>??*๐พอ=UฟwX?SพYฐำพ6-?wฟ;ฟ๔ ?wฃพ\JaพฃP?vฟร{>?๒O?(sBฟหBฟมรS?Z&?Sึฟเ$qฟ~LWฝด?\??a?4๔>?.ฟฦ?ฟdc9ฟ๖สฝi?@ol?ง็{?WะA?หต>ฮJษฝk๐ ฟวัLฟษvฟEJฟไ#lฟoDฟ'ชฟ|๚จพฑยฝv>!:ช>ใ0?๐ม'?ยฯE??ย\?้Bm?x? ~???ษ~?z?~t?ฌแl?ฬ d?"Z?P?jTF?  <?ฟ1?ฏ'?ิใ?Gm?&V?nฅ?Cพ๔>เ	ๅ>t-ึ>&ศ>๎บ>ฎ>Tิข>พแ>?>o>าv>ZOe>.ขU>?F>@W9>ึ,>ธ >7ฅ>คR>5ด>|๑=	สเ=7=ั=๗มย=ึEต=ทจ=๗=๕$=l=[*}=?k=hD[=L=yๆ==rน0=-v$==om=่=ษฏ๖<zๅ<Cกี<อฦ<K น<ซ(ฌ<25?<?z?Zๅu?"i]? ฟxู5พtษh>'ีผ>้ฟๅฝwพๅ?,ฐI?
3?-น_>W+ฟbฟืฅ?๏H	?}ฟ%i?ง3<ฟใ.?b[Gฟ'r?5บyฟS+?ๆั><0xฟO?R&?;)TฟุฟFlB?ม?K?ๅเซพgฅฟ,;ฟแ๐>8ณ}?ํน;?ด+;ื0ฟaฝ~ฟกXฟ Pฤพภl6>น?'?ฦQo?ฎย~?[(]?๖??R>iฝคนลพM)$ฟKใSฟ้งqฟCู~ฟ{ุ}ฟฬoqฟLl\ฟญ_AฟD"ฟๆฟHภพ.|พ.๔๚ฝRQฌป?๐ิ=ฎqQ>ฉซ>P%ย>ุR่>า9?ๆh?C?!?ย-?w;8?
pA??I?cP??ยV?L'\?ฐ?`?๕d?wh?กk?่Sn?+ฌp?ตr?zt?ๆv?ฒWw?Lx?ฅy?่]z?{?ล{?RV|?พำ|?k@}?}?'๐}?า6~?	t~?ฉ~??ึ~?ว?~?<!???ํX?Po?ณ?}?ข?ฎ?น?๙ย?'ห?<า?^ุ?ฎ??Gโ?Cๆ?ถ้?ณ์?I๏?๑?w๓?l5?จ?_้{?ฑ=>ฎVWฟojd?,mฟ{์ฟN1?|?Hทi?qฉr?ิาz?น้?>็@ฟRีฟ*x? Q?พuพผ?๐>4ฟ๖#โ>x]๔ฝWuอพล'g?โิgฟฐฟ=$c?บ*ฟlk(ฟณวA?ดn:?ถฟxฟพ?Q?พ\j?aO>X!ฟศ}ฟCฟ็ฃพXh๛>อพg?}?แ9H?%ว>U๎ฝฑร๓พP6HฟีฅtฟRฑฟknฟงHฟ8?ฟำ.ฒพน<ๆฝ็=sลข>ฝ	?>?)%?ควC?๋?[?ํ6l?ปrw?ฤ}?ง๙?ั~?;๙z?qu?ฤm?nำd?
Y[?]Q?G?4ษ<?2?q(?๑?"?ิ?#K?L๙๕>i4ๆ>Gื> 0ษ>้ป>็kฏ>>ฑฃ>ิฐ>?a>เป>kw>Yf>3สV>ึH>Y:>F->ฐ!>.v>b>ฉi>/ฮ๒=๐โ=~bา=ำร=$Dถ=Tคฉ=fใ=+๒=tย=?~=ๅๆl=x\=b-M=e๑>=?ฑ1=b]%=ฒใ=ฑ5=GE=ซ
๘<Lำๆ<ฏอึ<ฆไว<xบ<สญ<ก<4?s L?ธF6>ื{ฟSp
?ธ0็พX[?ถวNฟี8ฟ๑`=ไํะ>"ฃ>้แLพุ.gฟ
L(ฟ๕L?S }>:gฟa}?ฝ?aฟ3DU?ญeฟฯ+~?ใ{jฟt?>&*ู>ฆเ~ฟm)์>`โ>?"ฬ@ฟผI'ฟvp/?ANZ?mฃ|พ=}ฟฟฟIฮ>?ฑz??F?H
=๘ื%ฟd}ฟ0_ฟWูพลศ>9?`ืk?เw?ฒa?ณ?	ว>แฅHฝxบพrษฟฉไPฟP?oฟ`~ฟ๚[~ฟ?ทrฟD^ฟ๓Cฟt๑$ฟL(ฟ^ลพ/;พธํพdผ<ฤ=๒๋I>A>ฎฟ>๏ๅ>??๑P?ๆ ?z็,?({7?4ว@?์๎H?๕P?-QV?ขฤ[?ฏ`?ชd?IEh?hhk?ฅ"n?_p?Vr?ุYt?่ๆu?j?w?=jx?bmy?Nz??{?ฑน{?L|?ะส|?ฎ8}?฿}?X๊}?ส1~?ฎo~?Jฅ~?นำ~?๒๛~?ศ?๔<?W?ธm?R?K????บญ?ภธ?Mย?ส?ปั?๎ื?M??๔แ?๛ๅ?w้?}์?๏?^๑?T๓?oz?จ??ํE?yฤฟ0ใ|พ}ษ>ศุ=9ฏqฟv2M>lkd?n?เ}?ฃโO?#S=Ipฟฉพco?ฬr,ฟ๘=๏จn>ผฏพ8p>้eฏ=Oeฟ๘v?/วRฟฏๅvฝ ญq?เฐฟ๕?>ฟู6-?q,L?c?พสแ|ฟ^=eพฃbE?ธkq?ถe>Fฟ~ฟำLฟ+`Oพ๐eๅ>b?๒ู~?TN?(<ุ>sซ๙ผ'hๅพwnCฟ๊}rฟ฿ํฟฬpฟส|KฟO>ฟRปพ1พ่สล=ใF>Yค๗>"?ฮทA?~ตY?๕#k?วv?iq}?L๏?$?UX{?Wu?Ln?ํe?+\?4"R?ๅG?K=?นK3?d2)?[?Mื?&ฑ?๐??3๗>^็>yaุ>๙9ส>Nใผ>"Wฐ>ค>ฮ>ี#>?q>Aพx>Aษg>&๒W>(I>ษZ;>ซz.>ีx">G>ุ>>< ๔=ั?ใ=มำ=+ไฤ=oBท=ช=ิฟ=aฟ=z=ข๑=ํ1n=ฅฌ]=*LN=O??=Jช2=D&=฿บ=๒?=ฅ?=e๙<่<๚ื<9?ศ<ฅป<้ฎ<ี๗ก<FL\พ<
ผ\"ฟ;VPฟx?๑Wkฟ?~?ะจพถฮzฟะ,็พ]ฝฉพึeฟ๚จฟeฐพ?2t?๚ฝฯ3=ฟZ~?2๓xฟดp?*?xฟ6?+ดRฟนว>ผI?ีฟY?จ>ผ?S?๖w*ฟd<ฟn?[f?gUพฎyฟพ#ฟBQช>]xv?๏ฏP?S>?=เฟ	ฒzฟ๖?dฟ8N๎พTสม=ุ?yh?แ?h?d?[%?Eค>ฏฎภผถฏพฌUฟ{ฯMฟI@nฟ>ำ}ฟฅอ~ฟt๑sฟ4`ฟRวEฟ$\'ฟ.ฎฟPoสพk,พ\^พ๑ธผ6ด=OcB>ธิ>๛๙ป>แสโ>Gย?๓7?ะํ?,?บ6?@?JZH?้O?฿U?{a[?@/`?ิ^d?ฦh?n/k?๑m?YVp?๚jr?j9t?ลสu? 'w?Ux?[y?(>z?{?ผญ{?ฆA|?ีม|?ๆ0}?"}?ไ}?ป,~?Lk~?~ก~?pะ~?๙~?Q?ั:?<U?l?๎??๔?ำฌ?๙ท??ม??ษ?9ั?~ื?์??แ?ฒๅ?8้?F์?๋๎?5๑?0๓?Uxฑ>ฦ/?ิ<ทQ}ฟฟ๕>พ๒r/?pืvฟ\ฅดพp?QรY?*ืU? ?{+ซพ}๚ฟปU=ศ{m?้[ฟ'ล>?ผ์จไฝy<๒>ฺฤ6ฟ5?~?9๗7ฟ7ฦYพฉA{?ท}ไพRฟUW?-[?YฏพB}ฟ4YพL7?Pw?snฌ>ทkฟำฎ{ฟ?eTฟลพKุฮ>3๓\?_ฅ?สT?โ้>ึae;ใัึพL{>ฟs'pฟเ?ฟ๖rฟ?Nฟ๗oฟญdฤพ๙9พ๋ค=ๆพ>๐1๑>ญใ?T???ค#X?

j?"v?}?เ?u.?ัณ{?,v?Bบn?G2f?ั\?YๅR?XฌH?bX>?4?๓)? ?ฆ?^?ฌ?๑m๘>U่>{ู>Cห>`?ฝ>3Bฑ>ญjฅ>ฌN>ๅ>&>Mz>i>Y>C<J>{\<>k/>๓X#>
>ฮ>ิ>Br๕=ญzไ=?ฌิ=@๕ล=ท@ธ=ู}ซ=@==@=กช=๓|o=ยเ^=๑jO=9A=ตข3=ษ+'==3ฦ=บ=mภ๚<๎X้<&ู<อส<าผ<?ฎ<&ูข<!pฟNฟ๕ฟ๕อพ0`?)uฟ^l:?Vน>\oฟEVฟิฟๆฝฟ4f]ฟ EqฟูT<?ื?n.ฟพ5ฟkBl?j๓ฟ=f~?{๗ฟฬ:u?ญ@3ฟต=22?อ$zฟ6D>e?กฟA4Oฟtค?ดp?ิLฝqsฟแ`2ฟ(ช>q??Y?QC;>ค่ฟ๙ฃwฟา-jฟlKฟัUW=ปS?\๒c???`h?ห+? Vฑ>as ;งฃพฮฟคJฟ๛jlฟd1}ฟs-ฟuฟะaฟฐํGฟ;ภ)ฟx/	ฟ
zฯพ*พฬหพ~??ผ1คฃ=?ื:>f>Aแธ>ดเ>๊?๏?!๔?ฆ/+?๘5?Bs??๙ฤG?@O?bmU?ุ?Z?cุ_?,d?๎มg?*๖j?Xฟm?+p?mEr?ัt?{ฎu?vw?ศ?x?Hy?(.z?0๕z?ถก{?;7|?อธ|?)}?[}?ข?}?ค'~?ไf~?ญ~?!อ~?<๖~?ึ?ซ8?`S?j?~?โ?่?๋ซ?0ท?๒ภ?eษ?ทะ?ื???Kแ?hๅ?๙่?์?ป๎?๑?๓?>ฟธพS@ฟไJฟพ๕r?0Vฟw|?$ฟ Lฟ'.d=|w?ธฃ?ษ&ี=ะX+ฟ๐้mฟXrป>๛C?Hxxฟ?ห?Aภพzู?=3 Pพ๒?๓>๛|Uฟ๛~? ฟษUธพทฏ?
ฅพโcฟK๎๚>wih?ชๆ|พฟโฟูๅฦพ?ช(?Qo{?Q?อ>ซ?๏พlexฟD\ฟพอท>ธ?V?p??wY?๙>ธ}=ฃศพ็]9ฟโขmฟS็ฟ4qtฟ$Rฟฝ ฟ|dอพ&6)พn;=๐->ูฒ๊>5?M=?hV?3้h?`^u?8บ|?ูห?V?ญ|?๑v?ฅRo?z?f?]?งS?ๅqI?y??ิ4?ณ*?ั ???ด
?:?ง๙>ทฑ้>?ฺ>๊Lฬ>>ืพ>-ฒ>1Gฆ>o>>ง>ว?><d{>หBj>ำAZ>cPK> ^=>X[0>	9$>๎่>}]>๊>Aฤ๖=ตๅ=9าี=Rว=?>น=jฌ=ชx?=ฦY=?=q\=๘วp=?`=ทP="B= 4=?(=7i=s=`t=N?<พ๊<๑Rฺ<`+ห<?ฝ<&๑ฏ<wบฃ<iLฟจtฟ8P)ฟฤ(?OSก>ท5ฟ)>RD?%~ฟ;ฟeู\ฟP(\ฟ~ฟT3>ฟ|ฝ>:n?Oด$ฟvพ8ฬH?ฃkvฟ?}?ภฉzฟPa?Fฟ:ถฝ}ๅM?7ีnฟ?ฏJ=ๆ?r?ฝจ์พ%_ฟ8ซึ>ืx?Yพ๔<ิไkฟค@ฟq@>xrj?"b?vธw>ภพฟY่sฟ๏nฟฦ+ฟีำ*< ๅ?q_?๘ั?-จk?0?ฑkพ>?ศเ<๗พ4ฟฮbGฟjฟ
{|ฟ]{ฟเ7vฟ cฟ๔
Jฟ?,ฟฌฟก~ิพMพำ5#พE`#ฝs/=ฝI3>๖>ฦต>o:?>F ?็?{๙?bR*?>55?'ศ>?๙.G?๛N?ช๚T?ธZ?_?"วc?ภg?ผj?Om?ข?o?ญr?๘s?
u?ส๕v?b*x?6y?z?>็z?{?ม,|?บฏ|?6!}?}?บุ}?"~?vb~?ึ~?ฮษ~?[๓~?W?6?Q?โh?"}?ซ???ซ?fถ?Cภ?ฮศ?3ะ?ึ?(??๕เ?ๅ?น่?ื๋?๎?โ๐?้๒?FฟB9qฟ3`}ฟ๓ิฝใรh?6~ฟตb?ืภฝuฟงรๆพ?"1=?จ=๔ฮพDgฟOy<ฟ?ฤ'?๗
?]ชฟ}.O?ฟ@ฟzต>J6ำพ๊E%?dolฟ์v?ฅC่พกk?พ๋฿~?ทOGพs้oฟ@ึล>Yr?3พa~ฟๆ๔๎พ0?U~?๒ัํ>็$าพเ3tฟhcฟอดพ[R?>MWP????bณ^?	ื?๙;=/นพk4ฟณ๐jฟ<คฟC-vฟRQUฟ,?$ฟQึพ$;พษ@=y>k'ไ>>?ฮZ;?ู้T?zมg?V?t?8V|?มฒ?^z?่_|?ฃw?r่o?g?ุB^?2gT?86J?ใ??ั5?_r+?๓!?๓?๐ถ?฿?ญเ๚>ตฺ๊>*ญ?>Vอ>็ะฟ>ีณ>#ง>์>ัh>๐>ท|>mk>i[>sdL>บ_>>K1>%>หน>& >L?>9๘=T๐ๆ=o๗ึ=_ศ=>=บ=UWญ=Uก=๗&=พ=?=๛r=๘Ha=|จQ=
C=5=.๚(=c@=ณV=ฝ.=.v?<?๋<\?<๒Bฬ<+พ<Eใฐ<ศค<ถ๐=Rnซพ>>/~?๎!ีพC๑=eะํพ:฿~?ง๑ฎฝ dฟฐยฟต!ฟ฿5sฟOn?พ>-?$]A??{Yฟ๐RQ=F_?r๙\ฟ_o?ั3iฟ\C?oOฤพฮ"พซ$d??]ฟิะฟฝ{?uคฒพๆlฟฃ>r$}?u$๛=ฅhbฟlMฟC้=/ตb?qi?ฃ>V์พฦoฟ@ sฟ๚ยฟV?ฝฒ?>8ึZ?ธX?ณn?E๑5?ำ\ห>๗ฎX=kxพoฟDฟA~hฟ@ฐ{ฟ^ทฟณDwฟ-eฟLฟ=t.ฟ่#ฟ๐|ูพณ๊พ<,พจฯFฝCธ=น+>แ>ูฉฒ>oฺ>?>?ๆ?เ??9t)?q4?K>?LF?N?hT?5Z?_)_?ธzc?<=g?ภj?[m?๑ำo?บ๙q?ืs?tuu???v?฿x?\#y?ๅz?9ูz?y{?9"|?ฆ|?N}?ฐ|}?หา}?c~?^~?๛~?wฦ~?v๐~?ึ?W4??O?Bg?น{?r?ฬ?ช?ต?ฟ?5ศ?ฏฯ?)ึ?ล???เ?ิไ?y่??๋?[๎?ธ๐?ล๒?ืw๓พsืTฟwPฟฮI%?แdร>ฝb/ฟ๛Rู>๚>/rcฟภ$Vฟฬ?พินพีs,ฟ'ฅฟn[ไพn?^?ัูr>vแpฟ p??ศ4ฟ็r?ฟlชI?7ฦzฟwg?ชํพ๙: ฟึx?n?ฝqyฟ>ษ๔y?โๅJฝ/zฟม
ฟ๏?ฤอ?m?\gณพoฟม`iฟ_อพNv>`cI?M?{c?ฒฝ??ื=_ิฉพฉ.ฟkhฟฆ6ฟแลwฟcXฟื(ฟฌ)฿พ๓Mพพ๚<๊ๅy>??>๊ร?๋,9?BS?็f?	?s?์{?ม?๓?ฐ|?@xw?จ{p?d0h?ะ๘^?ใ%U?Q๙J?ขง@?3Z6?1,?TD"?ฆ?ฮb?D?S?>O์>รล?>ื^ฮ>[สภ>fด>า?ง>กบ>N*>F>ย	~>๗ปl>3\>txM>Fa?>?;2>๙%>ข>สโ>ซ๔>*h๙=+่=?ุ=j(ษ=~;ป=Dฎ=y1ข=&๔=}=ภ=?]s=}b=@วR=๑'D=๓6=`แ)==๓=้=ั?<^!ํ<ฦซ?<Zอ<Xฟ<cีฑ<}ฅ<64a?t??ฐX?}C?๒lฟ{:?Qหgฟึ__?ม๛๊>ผGฟๆ5gฟนnฟ?2=ฟdWฝ e?ล?>PฒxฟcYญ>,{ฑ>A5ฟฃZS?{nLฟKฑ?RุMพณ\?พ (t?ภงGฟPpพใ?ฉมkพ๑uฟt1]>ฏ?~v[>:Wฟต&YฟYW =c?Y?ลo?Hฺถ>coาพUsjฟcฝvฟ?ฟ;<ฝื๊>LืU??~?ถq?);?์&ุ>ฝj?=ฅมพกสฟ?@ฟ2gfฟัzฟrแฟBxฟ๕ษfฟฬ)Nฟ๙ร0ฟไฟัt?พ;อพา?5พ:;jฝฯ}d=ู%$>><ฏ>ทกื>'๛>ัษ?P?+(?+ญ3?ฏo=?๐ F?M?T?ะY?8ั^?์-c?d๚f?Hj?y(m?จo?ำq??ตs?ธXu?ฤv???w?y?ฃ?y?!หz?A}{?ฃ|?m|?[}?อu}?ำฬ}?8~?Y~?~?ร~?ํ~?P?(2?ผM?e?Nz?8?ผ?,ฉ?ฮด?โพ?ว?+ฯ?ถี?a??Jเ?ไ?8่?h๋?+๎?๐??๒?su๗>Qฝๆy>ฒ}?ณพ5๛hฝฏzVพPิe?Oวฟ์ฟiQMฟฌ?;ฟษjฟข]qฟ0ฦฝถ|?$ฝ๙bMฟ-?ข\ฟั>?หCฟe?ะ๛ฟ๑O?ฦพW๕<ฟ=ณm?ำ็=~ฟ๘(>r~?N=฿sฟ ตฟิP๋>๎ี?ปJ?5่พ?(iฟ9์nฟ?ๅพู`>B?๒F~??้g?Pt?P>yพๅ)ฟeฟฃ~ฟี:yฟปZ[ฟM,ฟcํ็พญา^พj-e<ถj>ํ์ึ>ฎ ?ผ๗6??Q?]e?}s?q}{?ูq?Iต?w?|?ศ๋w?Eq?ึh?๙ฌ_?ใU?.ปK?ฒjA?ม7??๎,?'?"?X?O?4'?}Q?>+ํ>??>jgฯ>รม>ห์ด>๎?จ>>ณ๋>๛>Y\>i๘m>ฦธ]>eN>ฦb@>,3>ู&>s[>iฅ>ช>บ๚=ไe้=ฬAู=p9ส=บ9ผ=ศ0ฏ=?ฃ=Sม=<=ูq=?จt=)ฑc=ๆS=ุ2E=[7=ศ*=ธ๎=3็=vฃ	=๖ =-d๎<1ุ?<rฮ<ภ<วฒ<j^ฆ<(`?ช?X:t?
=Cฝoฟๆ?VRzฟWแ>ช\?Q?gฝ)่ฟp้-ฟWฬพกช>ห~?จฯ3>ใ>ฟะ?๗ฃ=พิฟO+?๒พ%ฟฎโ>49ผฏQฟ?}?=ท,ฟ๚พพภ[?g4?ฝ}|ฟ,]แ=ฃฌ?Kบ>nJฟ1\cฟ:พฝ๖O?เu?กfำ>b?ทพภdฟรฤyฟํ'ฟ๕3ํฝ๔ฯุ>aP?}?t?t8@?วไ>Tcิ=ฏ๐iพล๛ฟJ=ฟ:dฟ?yฟ๙ฟA1yฟHZhฟ0+Pฟฝ3ฟ๐ฟfใพวซ?พ^]?พ&ัฝJC=N>9>บjฌ>Rาิ>)
๙>วซ?ฯ?;ต'?็็2?Tย<?่hE??L?@S?rjY?คx^?ภเb?6ทf?*j?ซ๕l?แ{o??ญq?ตs?ี;u??ชv?้w?ม?x?Kํy?๖ผz?๙p{??|?4|?]	}?แn}?ิฦ}?~?U~?4~?บฟ~??๊~?ศ?๖/?ีK?๙c?แx???ช?>จ?ด?0พ?ว?ฅฮ?Cี??ฺ?๓฿??ไ?๗็?0๋?๚ํ?d๐?|๒?ฦo?9:E?ภk?<?ูeฟ!?์Bฟฬ1}?ปw๛<?dฟๅ }ฟ	uฟ๛ฟrd>ฟ>|?Zฤพ<ฟw6{?์<vฟxa?1ฆbฟาฺw?ข฿{ฟั?2?ฬญั<?6Uฟ{ฐ]?ึO>ฟ๗ฟ็M=พ??iy>ฑkฟู?-ฟฮ๒ล>๗m~?ัm#?โgพบYbฟRถsฟจx?พสจ/>ุA:?อ|?
?k?B๘?	ฏ0>A๖พKY#ฟหอaฟL?}ฟ่zฟ6^ฟ#b0ฟk๐พ^pพแย'ปE3[>>ะ>ฉ6?Vป4?ห?O?X!d?น@r?ฏ{?
J?_ฬ?ศF}?8\x?Hq?yi?S_`?ศV?ฯ{L?พ,B?y?7?Dฌ-?lต#?ญ
?pน?ืส?*?>VS๎>๖?>ปoะ>ขผย>ืต>็ทฉ>bW>ญ>?ฏ>iW>ร4o>Fเ^>G?O>:dA>94>น'>=,>h>Z_>๘?=ค?๊=๔fฺ=sJห=๔7ฝ=~ฐ=@๊ฃ==๛=ค#=๛๓u=?ๅd=ลU=ฝ=F=ร|8=ยฏ+=โล=rฏ=า]
=fร =?ฆ๏<฿<ฉฯ<ฐ!ม<นณ<บ?ง<ไห=ก5#?Dว>$-ฟWzไพ"2M?r๒&ฟ/พแ?rZๆ>7พๆ9พฃ@<;"+?ป?v?f|&พq{lฟัM?ใEพw
พ๔>๎พ >?7>eฟ7ฟ๊?Iฐฟั ฟุฎz?ย<$ญฟ	ป7;ฟ}?\`ศ>ฦ<ฟ๊lฟTโฝญE?/^y?i-๏>ธพ}m^ฟ4|ฟ?/ฟ?_!พึrฦ>F K?)|?dv?ฑE?8<๑>O>f<Rพ๊8?พ9ฟซ๘aฟํีxฟษ?ฟ฿zฟc?iฟ#RฟrN5ฟ๘mฟณP่พ6ฅพญทHพฝ"=๘>ใKx>\Hฉ>๏ า>'๖>ภ?\?iิ&?ำ!2?;<?3ะD?ฮwL?\*S?cY?ฃ^?4b?ณsf?pำi?ยl?Oo?ทq?Ass?ฬu?ฮv?ฉำw?ฮ๊x???y?ธฎz?กd{?L|?๎|?T}?๋g}?ฬภ}?ฬ~?P~?H~?Uผ~?ฏ็~?<?ม-?์I?Qb?rw?พ??Pง?3ณ?~ฝ?gฦ?ฮ?ฯิ?ฺ?฿?๓ใ?ถ็?๗๊?ษํ?:๐?W๒?ํK?โดx?ฐภd?ณต<Quฟ#y?	ฟแ:?6?yฟb?qฟชภ|ฟ6Thฟา๒?พ|?ฃF`?๐ธ&ฟR?ซพฤd?ำ๏ฟฤgw?4wฟีต?๚nฟP๏?=zB>lhฟK"I?ฉ>}ฟผhฝj~?@ศ}>s@aฟu?=ฟไK>ๅ{??ล0?r[&พทZฟ.ปwฟL2
ฟ<ฐ?=02?&เz?ณo?๗F#?ิูR>ทvพ{zฟงj^ฟร๏|ฟ๊ธ{ฟว๖`ฟ๏'4ฟ๖2๙พKพ๕ผลK>>ษ>?e?ัw2?N?m?b?มiq?]z?T?4฿?t}?ษx?ฏ%r?๘j??a?๚XW?2;M?ฤํB?Z8??h.?"m$?Rผ?4d?-n?Zภ?>ยz๏>ฅเ>สwั>vตร>มถ>ฝช>%>:n>?d> >qp>ณ`>ดP>กeB>X5>?(> ?>*>ฌ>ิ]?=_?๋=?=r[ฬ=*6พ=1
ฑ=?ฦค=ช[=บ=oี=๘>w=Tf=#V=ขHG=*u9=๓,= =ฑw=.=ีp=ห้๐<1เ<;กะ<?%ย<ฝซด<!จ<gลMฟชฏrพฺๅพแ๏ฟก>}k>ฦyฝ๖.ฟTIT?rV?มง>ื">+ื>๏f?K^N?๔?๖พUDBฟฮq?ภๆพ4?ตผษ>ผพะB=๓น>aTฟ9W{?รึพศ^ฟlq?ฆ?>ฮtฟq๒ีฝpx?J๓>f,ฟ]Ksฟะ-=พด+9?b|?U
?๊พ~Wฟ?~ฟ๏แ7ฟH?Kพวณ>เ+E?ฃz?แvx?<ีI??>?๐>ูi:พZ๔พ๊เ5ฟก_ฟบwฟ	๔ฟNแzฟ4VkฟsTฟ7ฟๅัฟj4ํพh\ชพRพ0ชฝ=^>?[q>)$ฆ>-ฯ>%?๓>ภl
?๚?ท๒%?๏Z1?ce;?ำ6D?~๑K?๎ดR?ฺX?4ฦ]?GEb??/f?ji?Ll?์"o??_q?กQs?u?|xv?ฒฝw?ยืx?Yฬy?g?z?8X{?๗{?|?A๙|?์`}?ผบ}?~?๔K~?W~?๋ธ~?บไ~?ฌ
?+? H?ง`?v???aฆ?cฒ?สผ?ฬล?อ?Zิ?4ฺ?D฿?จใ?t็?พ๊?ํ?๐?2๒?{บตพoิว>ฌr>Uด4ฟ4ฟkบd? ๔Tฟ}Y>+:j?okฝ.ฟ]rQฟ3๘'ฟ-\ฝต1W?ชN*??ZฟฬศEฝTฯ<?๗yฟ๏?ฃลฟา|??Xฟy9ฮ>@ทฒ>"vฟt0?cF่>h wฟ*?7พ'Cz?๙ฦฏ>7๙Tฟ?rKฟ8o>ฮWw?f@=?๖เศฝ๗GRฟ๗zฟiGฟu=๊)?x??
s?๐]*?Kฦt>ษWพรxฟา?Zฟ.ู{ฟฏม|ฟ่cฟIฺ7ฟู ฟsๆพJฝgK<>Uมย>ว?B-0?.[L?ฮa?p?}z?ธ๋~?ศํ?zฮ}?ฮ3y?zฎr? บj?พa?ฉX?V๙M?ลญC?e[9?ศ$/?H$%?m??6	?{ ?ศก๐>๗$แ>า>ฎฤ>๘ชท>ooซ>ฒ๓>[/>ฉ>ตฉ>.ญq>/a>?วQ>๛fC>m?5>เx)>ผอ>$ํ>๙ษ>ชฏ?=ํ=7ฑ?=nlอ=^4ฟ=โ๖ฑ=?ขฅ=ำ(=y=8=๓x=hMg=FBW=SH=m:="~-=5t!=๏?=า=C=,๒<m]แ<ฬธั<*ร<?ต<[ฉ<๏ToฟfฒkฟตxฟmU5ฟ)j\?๓ๅพ?ว(zฟV?ฮ>?MO;?f3?0A??ก?ลฉ
?ศ!?ฟLษฟณ?สเ,ฟGl>ฑบ<สำ,ฝณใ พศi?ิPjฟ๓ๅo?งฦพx9ฟ๘#d?ข>[ี{ฟmWพฟpp?โ?bฟ๊xฟพ๛d,??ร~???Jพ}๘OฟยGฟ!ร?ฟ๊๛uพLี?>,??!x? Jz?
aN?(ฬ?jซ7>w|"พิ]๊พ$2ฟฆ5]ฟ2vฟXึฟข{ฟฅมlฟ%๖UฟUผ9ฟค0ฟ๒พ=.ฏพผ^[พ฿ฺปฝู$ม<ย>Zhj>,?ข>MXฬ>+u๑>ฦK	?ช?%%?<0?ฯต:?ฦC?jK?๖>R?ี6X?Xl]?๙๖a?ฏ๋e?]i?บ[l?๖n?9q?ี/s?Iไt?	_v?งw?ฤx?ภปy?z?ฟK{?ป์{?>x|?#๑|?ใY}?คด}?D~?aG~?a~?}ต~?ยแ~??O)?F?๛^?t?>?l?pฅ?ฑ?ผ?/ล?อ?ๅำ?ฮู?์??[ใ?2็?๊?fํ?ไ๏?๒?!|zฟKUฟn#ฟ๔ฟูษ[>yรส> cงพiษยพ฿=}?0๑ๅ>ไ!พน๗พe\พช>?z?ธม>งOyฟVy>ุ?M/bฟ=#z?๏{ฟ~No?Iฦ:ฟy&p>aO?>Z~ฟย(?wp?ผ๔lฟDkพล	s?5่?>/ฺFฟ4 Xฟ๕K>ฮณq?ธฯH?dฝ]Iฟษh}ฟๅ๓ฟฅvื<ผ ?สฐu?`v?ม:1?45>อP7พwUฟ๙$Wฟปzฟฆ}ฟช"fฟษx;ฟฎฟqฅพ^ปUฝฺล,>0๓ป>-ฑ	?ม?-?แJ?D`?Mฉo?y?7ต~?๘?ุ~?๐y?ฅ4s?Wk?vkb?ีศX?:ถN?พlD?:? เ/??ฺ%?>?ธ?๑ณ	???gศ๑>๕;โ>ำ>zฆล>ฏธ>?Jฌ>ฏม?>d๐>cฮ>ฤR>?้r>QVb>?R>HhD>w์6>ผX*>r>ญฏ>B	>ผ  >รP๎=Qึ?=f}ฮ=2ภ=ใฒ=\ฆ=๚๕=8=9=ํิy={h=aX=i^I=๖e;=Qe.=^K"=-=ๅ=ฒห=go๓<ึโ<^ะา<2.ฤ<๘ถ<ซใฉ<XhSพlํ\ฟ฿YOฟฬ:<z?ิkฟฟ๎q??ฟlฟ-๊พปd?ณ1w?{
`?Qu?vq?คฏR>๎๗lฟชhพKx?+Yฟพอ๏>๓โYพศ5>c?ตพ6/?B?xฟผๆ]?wj?ฝลPฟwฌR?+ศม>'?tฟwฺ?พ	wf?ะ9!?แ4	ฟ5์|ฟ_ถจพoว?จ??wp?ำOพถเGฟT้ฟ?Gฟหีพฒฃ>?ป8?%[v?N?{?ฟR?ฝ
?OGQ>ฑw
พึCเพT.ฟตZฟXFuฟธฆฟpT|ฟค nฟัWฟV่;ฟฟจๆ๖พ๛ณพซdพอฝวH~<\I?=)rc>lึ>ษ>;๊๎>ื)?n?ต,$?ปส/?~:?C?ใJ?tศQ?VฯW?]?Lจa?.งe?!i?ๆ'l?ษn?๑q??s?ฮฦt?tEv?nw?_ฑx?ซy?z?5?{??แ{?ิn|?๚่|?ัR}?ฎ}?๖?}?ศB~?f~~?
ฒ~?ล?~??'?!D?M]?s?๛?U?~ค?มฐ?`ป?ฤ?ฬ?oำ?hู???ใ?๐ๆ?L๊?4ํ?น๏?่๑?ฯ3ฟo~ฟ8ฟฉห-ฟR?โนพF>ฉVฟ'CB?ไU?bc>_Tฝ-l>๋*?=ะ|?#^E=eฟเซ?ใX>ฮง<ฟCWf?เkฟผW?4ฟ?q=ใv"?ใฟูฆ้>(,?!$_ฟlูฬพi?ฃ็?ฅ7ฟล.cฟฐ=ดj?อdS?=ื$?ฟ๐ฟF0*ฟkทผ?นpr?)x??7??>ไ^พ๔ฟำCSฟ?.yฟ๒e~ฟปhฟ?ฟR5	ฟXพปฑฝั5>)ต>Nอ?f+?ชฝH?ํ^??ฟn??x?ำy~?-??G~?๖?y?1ธs?ฺ๑k?c?{~Y??qO?ฎ*E?๒ึ:?0?ใ&?ฮ??b?^V
?{ฑ?๎๒>กRใ>dิ>ชฦ>:~น>j&ญ>ก>Wฑ>
>ฤ๛>8%t>}c>/๏S>iE>w?7>8+>!o>1r>4
>?ฉ >l๏=g๛?=Zฯ=ผ0ม==ะณ=ท[ง= ร=|๗=ศ๊=ๅ{=ตi=ยY=KiJ=\^<=L/="#=kะ=@G=!y=4ฒ๔<?ถใ<๏็ำ<]2ล<ท<?ฤช<B86?๖โฝ	ุฝ9๗;?ุ?JCuฟSLs?|ฟPบ&ฟ>ช??y?ฺง?C๊}?>?ง!พเณฟ]7ฒ=บ[?:่uฟ๊-?mk฿พ๚ฦ>Fั	ฟ๏\O?\ฟvืE?t,๑<์ษcฟPw=?R๓๚>๛jฟ'ิพิ/Z?3? ์พIฟ[tฬพd?7฿?<K*?้8ฐฝู<?ฟ๏ฟ6SNฟ(lคพtt>C#2?Vูs?0}?o๎V?B?jภj>๘ฝไฝไึพ"q*ฟํXฟฃ๎sฟ+eฟ๗|ฟsoฟ:ขYฟ๎>ฟB?ฟ็ด๛พUฤธพW๒mพ9%฿ฝB๔;๋	ํ="y\>๒ฌ>จฦ>]]์>๓?G?hH#?n/?rT9?ฎfB?๕ZJ?hQQ?\gW?[ท\??Ya?Xbe?ๅh?า๓k?ฯn??๊p?ป๋r?-ฉt?ฟ+v?!{w?x?Ky?uz?2{?๑ึ{?\e|?วเ|?ถK}?\จ}?ก๘}?*>~?fz~?ฎ~?ฤ?~?้?ั$?.B?[?ฃq?ท?<?ฃ?๎ฏ?ฉบ?๔ร? ฬ?๘า?ู?;??ยโ?ญๆ?๊?ํ?๏?ย๑?นธ`>xง.ฟึ4ฟส๓s=LM}?ฯUฟ[R?.ๆฟK่>%?ฏ
'?!?ว>ะญ?ๅฯf?ร]?V3พ?tkฟ%ฐ=?N<๑
ฟำฆE?paPฟํ7?๕ุพS?๒ฝฟี@?yด{ฟ๘1ฆ>4าC??ไMฟฦ ฟ@K\?ฉ?ฐ%ฟ{lฟธFรปbb?u๒\?vxล=ฎ4ฟฌโฟ_๕3ฟwฝ?Nยn?๓อz?ค<>?๑Xฌ>y๎ฝฏฟ:Oฟwฟ2ฟฯ?jฟฒxBฟ"Pฟ?ฃพ๊{ฎฝ<>9ฎ>Lใ?H$)?ไF?]?Nะm?ฅmx?9~????~~?฿_z?9t?hl?บฟc?2Z?=,P?็E?r;?]T1?VF'?Q~??|๘
?L?p๔>๘hไ>eี>คว>gบ>ฑฎ>R]ข>2r>7>ตค>au>?คd>มU>ผjF>mฬ8>\,>ษ? >ฏ4>ฦ้
>R>ฦ๐=x เ=Kะ=็.ย=็ผด=8จ=D=uถ==?j|=้j=Z=,tK=ภV==ฎ30=ฏ๙#=จ==&=๕๕<จโไ<?ิ<6ฦ<3tธ<Lฆซ<iยy?+;?สu5?ฉ??Fพ.สฟ๐?ง๎ๅ<ต?tฟ5o=?A?ฤฉk?GSZ?Cw?>ฉษ?พ1uฟ็๖ศ>?้+?๕?ฟฝW?ฒ"ฟฦด?ฦ้2ฟh?1~ฟi`(?ฒษ:>Q#rฟ๊ไ$?9!?ร@]ฟฦhฟบK?ะ<C?ัรพึ?ฟ&๏พฑL?ฅฮ~??5?+๕ผ๒6ฟyZฟq๚TฟฎทธพSAM>yN+?sq?่A~?!๎Z?=O?@	>?kดฝBฝหพู{&ฟขvUฟ/rฟธฟN}ฟนpฟsi[ฟ	*@ฟ๙,!ฟ?= ฟXฝพU4wพ$ฤ๐ฝฯ8บว?=[}U>ว>อร>ฮ้>ใ?7???c"?U7.?ชข8?ขสA?>าI?ำูP?่?V?9\\?ั	a?.e?lฉh?|ฟk?Tnn?รp?mษr?ft?่v?ทdw?x?py?efz?๑%{?๗ห{?ู[|?ุ|?D}?,ข}?D๓}?9~?`v~?ซ~?ฟุ~?L ?"?8@?๊Y?+p?p?!?ข?ฏ?๒น?Uร?wห?า?ุ?โ??tโ?jๆ?ุ้?ะ์?c๏?๑?p?7>ฬฉ=ฌ้B?o ??K~ฟถฟ?PNฟร/พsีd?gn?ฏA?คTS?B?๔w!?FฟA@ฟeg?พว?พQๆ?ว*ฟล?ะwพ,ธพู๏Y?Wqฟวฉ>>ฆX?3{9ฟ4iฟJ	M?+.?๊นฟctฟ)ฑฝAหX?ษle?โช$>ำ7)ฟO้ฟW<=ฟ
ผ๔ฝิ??6งj?โ|?E]D?ิ?ผ>ฎฝธ_?พคKฟh?uฟฝwฟmฟYูEฟผ]ฟbฌพ?9ะฝ๔๛=็Nง>I๓ ?พ&?ผE?,\?ชฺl?ซืw?d๔}????ฑ~?จฝz?bทt?ม m?gd?4ๅZ?ZๅP?rฃF?O<?2?7๛'?จ-?eด?K?qๆ?ฺ9๕>๛~ๅ>"ึ>fศ>หPป>ี?ฎ>๘*ฃ>๕2> ์>M>฿v>ชหe>CV>ใkG>Xผ9>๘,>k!>(๗>>\๛>ฎ ๒=Eแ=8ฐั=-ร=ฉต=fฉ=g]=mu=TN=ฯต}=ชl=:ฝ[=L=$O>=?1=ึะ$=ๅ`=๕ป=?ำ=ฮ7๗<ๆ<ึ<ณ:ว<Pfน<ฌ<รWฏ>(เ{?^?๗%?AฤGฟ์=๊a5ฝzO?ี/wฟโๅพูบ>`(?ญ{?ฑย`=ฝงFฟUฃNฟน(?(?>๙?vฟms?-Lฟ:ถ>?}Tฟ๊cx?hฯtฟฌO?ฉ>๗{ฟ๗d	?ำq0?๑LฟัHฟ<;?วQ?d+พบ
ฟ?%ฟu(ใ>ซ|?ำ@??ื<oi,ฟc*~ฟบ1[ฟ ฏฬพ4ฝ%>6?$?N?m?ธ?Kฝ^?]ํ?ฌ>P?ฝ;Sมพt"ฟdนRฟqฟcฌ~ฟ+~ฟ>๒qฟฏ&]ฟ?Bฟ/v#ฟwฟGยพl8พ8/พสผฮ=๊~N>๔T>R๐ภ>่=็>Vพ??๛?<}!?pl-?(๐7?ํ-A?๐HI?ตaP?๚V?ซ \?บ`?ฏืd?๒lh?ๅk?@n?ip?๓ฆr?ymt?๑๗u?0Nw?wx?xy?ตWz?6{?๎ภ{?IR|??ะ|?d=}?๔}?แํ}?ฺ4~?Ur~?ง~?ที~?ซ?~?F ??>?5X?ฐn?)???ก?Fฎ?9น?ถย?์ส?	า?2ุ???'โ?'ๆ?้?์?7๏?w๑?๏นK?วUe?ฅP?oต~?!	ฝ๓ใ/ฟึD?๗่งพy!<ฟt??iV~?7/w?๔{?kq?Fย>IZSฟ?ฟ6}?ตฟคฝHห>นศ๙พ}โภ>ฎ^Zฝ้๗้พm?ฃดaฟ4=4[h?%7"ฟภ 0ฟวi;?g@?52?พ๒?yฟใW*พ?M?9ษl?ิ็e>ำNฟี ฟฎ?Eฟ๖ต+พ4^๔>O!f?=~??:J?๖ฐฬ>L[ฝE(๏พ0ฐFฟู๘sฟษฟ?oฟค$Iฟภ]ฟฎ#ตพ9้๑ฝv??=g[?>ฯ๚๛>#R$?'C?uมZ?๕?k?2<w?[ช}?ู๖?ทแ~?Q{?3u?โดm?e?D[?2Q?D^G?แ	=?ํล2?ฏ(???ๆ\?ฬ;??ฺ^๖>ชๆ>ขื>๑ษ>ะ9ผ>ิทฏ>๘ฃ>ก๓>?>i๖>ุw>๒f>ต)W>?lH>8ฌ:>ฺื->แ!>น>8T>5ค>F;๓=jโ=!มา=3+ฤ=3ถ=ป๐ฉ=*=d4= =ย =ธQm=๕?\=์M=G?=2=?ง%=")=Pv=j=z๘<w;็<?.ื<?>ศ<mXบ<๋hญ<rฟRใ>( ?ไpอฝ*Kฟย:?ลญ#ฟืq?์-ฟณรUฟF้ฝU4>N5>ฉพ๕7sฟjsฟ!;\?ผ>ะ[ฟญy?	kฟำw_?#mฟฬข?ตcฟ'ม>O๑>ภฟ๏่ึ>ฝF?ๆ9ฟ'd.ฟษ?(?^?แบ^พ?n|ฟ?ฟxย>เvy?E๗I?พ?จ=G"ฟ	`|ฟอ๕`ฟgIเพ๏๛=โ๗?ฯฆj?=ข?[b?m!?d>??&ฝัถพอ[ฟs่Oฟqoฟ35~ฟ~ฟยsฟูู^ฟoMDฟะน%ฟ7๙ฟฐวพึำพใ๙	พ็ผ8ฟ=็}G>&>ภพ>\ซไ>ก?a๖?` ?ย?,?ํ<7?@?ฟH?้O?,V?ฑค[?ืi`??d?/0h?Vk?ฑn?sp?Mr?fOt?ุ?u?7w?kcx?ygy?๓Hz?j{?ืต{?ญH|?๋ว|?-6}?ด}?v่}?)0~?Dn~?ค~?ชา~?๛~???D<?}V?4m?฿?่?ฉ??pญ?ธ?ย?aส?ั?สื?.??ูแ?ใๅ?c้?k์?๏?Q๑?๚ขฝ;d??hx?\??ญ็;ฟtฝ?0g>หฅ>5บ{ฟโr=b$S?e{?~ox?~ฦ>?3็Cฝjwฟ$9^พจA}?ฤ:ฟd0B>๐->ๆพ๑=8>>Gkฟ"มy?rLฟหฝt?mtฟ@Dฟม?'?์LP?ฐฟาพภ}ฟ+{พrB??r?V>ฮาฟ?}ฟ@6Nฟฏฅ\พOผ฿>จ2a?t?pำO??>ฌ0ณผฮปแพ1Bฟป๊qฟt๖ฟMqฟ8ZLฟอOฟQ?ฝพฯร	พk?ฝ=x_>๖>M฿!?่0A?kPY?7?j?>v?t[}?ๅ๋?ฤ?ุo{?ฌu?สFn?Iฐe?ษE\?ฤSR?	H?ฯร=?ง}3?@c)?๎?????e?r๗>ช็>ฯจุ>D}ส>จ"ฝ>ฏฐ>์ลค>5ด>้T>,>"y>h>=X>nI>;>ท.>ฑ">	|>k	>M>ุu๔=ใ=าำ=U)ล=ีท=อช=จ๗=Y๓=?ฑ=ฺ%=รn=ฎ๚]=หN=้?@=5้2=$&=^๑=ฉ0=ุ.=gฝ๙<฿g่<1Fุ<Cษ<Jป<;Jฎ<)0ฟฎ๋็พ	wพธIฟx?-ฟ??cyฟะถv?้M:พฎฟ(ยฟR[Aพwพ็ด*ฟ ตฟ;|พ*ๆy?ฐพ2ึ/ฟหZ{?G|ฟแbu?}า{ฟ%}?JlKฟpฯ`>฿??ฝ~ฟ7>๊X?b$ฟq|Aฟั?ฃNi?ซsพ/xฟD'ฟ๘ก>q5u?๗S?ดจ>?ฒฟ?yฟขCfฟ}๓พNฌ=๚z?๔	g?R๐?ณฦe?ฤฮ&?Tzง>คาผ/8ฌพ๊1ฟMฟหmฟ1ฌ}ฟ็~ฟ~>tฟ฿`ฟSFฟส๗'ฟ	QฟศถหพOlพ่มพซ็สผ<์ฏ=gz@>y๖>g1ป>๖โ> r?๐?ซฎ?Jิ+?๘6?๒??4H?แoO?ฐยU?KH[?K`?ดKd?!๓g?๔ k?ไm?kKp?|ar?-1t?รu?ฬ w?ฐOx?\Vy?:z??z?ฒช{??|?ฟ|?์.}?k}?ใ}?q+~?/j~??~?ฯ~?`๘~?ฐ?F:?ฤT?ตk??ษ?ฐ?ฌ?ฦท?tม?ึษ?ั?aื?ำ??แ?ๅ?(้?8์?฿๎?*๑?ฝaฟี_->ูใ>?Gพฅ?ฟ??ใัพgD?๊หmฟๅพฅทํ>ข?L??cI?ข๛?>Yอพฺ-ฟตNว=Dg?Iญbฟฆฎ?>ด๛ฝ`ฟฝฑ8ศผขZฅ>ป๓<ฟล?6w2ฟIvพv|?์0ูพGโUฟช่?์^?Nzฆพwบฟฅฅพฌ๕4?(x?ฒ>rฯฟถ%{ฟJ?Uฟพพัคส>?[?พ?%U?์>:ฒ<Mิพท=ฟfดoฟ?ฟฑ์rฟฝyOฟ3ฟฦพYพใา=y[>$๐>f?=??๖ุW?uีi?ำ๔u?ฐ}?ฐ??%6?=ฤ{?W"v?wึn?Rf?ร๓\?	S?มะH?฿|>?ฌ44?g*??8 ?มฌ??}??ณ?กง๘>ฟ่>พฎู>_tห>Sพ>emฑ>:ฅ>ฑt>1	>฿G>Oz>N@i>hPY>oJ>ู<>5/>&#>q>>พ>?๕>eฐ๕=ดไ=็โิ=t'ฦ=uoธ=_ฉซ=ลฤ=Mฒ=c=Rห=ฮนo=f_=ฉO=J8A=`ะ3=JV'=น=๋=E?=2 ๛<F้<ภ]ู<2Gส<ฆ<ผ<+ฏ<ํx๑พึM|ฟrOeฟ5r}ฟ?X;"M?"iฟ|3$?\ฝ>ื๏dฟผbฟะฟ'ฟฐfฟByjฟเฃ=์ศ~?ctฺพ๐ํพ้?f?`oฟ@g?l์ฟ)Hr? น,ฟฑฑd=7?ขxฟซ)>Lำg?ญฟJYRฟ#?>ๅr?Yข<ฝ์Srฟ95ฟcญ}>d๋o?au[?coF>ดฟ?wฟokฟz!ฟ7=ห?อ(c?ใ??`?h?๔,?lภณ>Rปู;ฬกพa๗ฟJฟLlฟg}ฟ=ฟcQuฟซ!bฟแQHฟ0*ฟฺค	ฟจfะพปพพ-rฝ๏?=t9>โฤ>NOธ>ผ฿>uJ?๛้? ฦ?
+?Kิ5??S??sฉG?*๖N?UXU?y๋Z?_ศ_?9d?สตg?๋j?*ถm?!#p?>r?ฮt?Cฉu?๏	w??;x?*Ey?4+z?ข๒z?~{?O5|?#ท|?ข'}?}??}?ด&~?f~?๙~?ฬ~?ต๕~?a?F8?S?5j?G~?ฉ?ท?ภซ?ท?าภ?Iษ?ะ?๘ึ?x??;แ?[ๅ?ํ่?์?ฒ๎?๑?i_ฟ!r0ฟ3?ษพัOฟ.ม:ฟk๚x?ฐ`ฟ๊~?๔?ฟBฃUฟlปข๏้>.๎>ผWe=์ต3ฟ7ยiฟ๔าอ>!ๅ=?Ygzฟ!%??พ1p>แ>iพa?>Xฟคv~?ณ3ฟนธภพ??แ!พซdฟI๕>ศถi?Aqพ0สฟ๊หพโ&?จึ{?๙\ั>ฬก์พK๖wฟp๎\ฟ?jพะ"ต>E$V?๙??๒-Z?ญc๛>;g)=วOฦพuล8ฟ9Vmฟอแฟฬฅtฟ?Rฟ!ฟฺjฮพD+พน|=ษO>ศ๓้>ๆ?ฒB=?[V?ทวh?๓Hu?ฏ|?:ษ?ูZ?|?v?๊co?
๒f?1?]?ฝS?kI?5??๛๊4?๙ศ*?Qๆ ?T?m?VM?fห๙>ถำ้>gดฺ>Bkฬ>ะ๓พ>๖Gฒ>i`ฆ>5>eฝ>๐>{>gj>ฉcZ>๙oK>{=>ึv0>ฌR$>ำ >รs>ช>๋๊๖=ูๅ=ล๓ี=%ว=\น=ฎฌ=โ?=@q=_=ศp=ืํp=8`=ชP=ซ0B=ท4=o-(=ี=\ฅ=ฒ=?B?<ญภ๊<Puฺ<\Kห<ร.ฝ<ูฐ<ชp๙>้z9ฟZCkฟฅ}ฟจ/?70n>ซ\๕พ็:ผ=ย?N?ขฟq?ฟ>ทcฟณกbฟ0ฟQ6ฟ๒ผะ>\gj?vฏ+ฟ?๐TพนC?Ktฟ}??Eyฟ+^? ฟ๗เฝPฝP?,mฟ?=๓{s?0ๆพไศ`ฟ"๐ะ>?ฦx?เ๓%=ฃๅjฟ? Bฟนโ7>mi?ฑc?x~>SฟpsฟซqoฟxHฟธท;ถ๊?_?ํว?hl?01?ฃ่ฟ>wจ๘<FวพฉฌฟGฟ๕Ejฟเd|ฟ#ฟcWvฟ-ถcฟKHJฟub,ฟ๔ฟ3ีพพSI$พHn'ฝQM=Kl2>ฦ>|kต>ฒ่?>" ?uโ?พ??9*?็5?ด>?ลG?์{N?ํT?<Z?w_?iพc?(xg? ถj?m?ฅ๚o?Xr?I๔s?วu?๕๒v?ํ'x?โ3y?9z?ฅๅz?<{?+|?ฏฎ|?O }?ย}?ุ}?๑!~?๓a~?f~?mษ~?๓~??C6?JQ?ณh?๙|??ผ?็ช?Nถ?/ภ?ผศ?$ะ?ึ???์เ?ๅ?ฑ่?ั๋?๎??๐?ข}ฝ '~ฟbtฟย{ฟฝ๘	e?ใ#}ฟา#_?Xฝ(ฟu๑พ	๏ฎ<ผฮ}=kฉพ๘hฟY9ฟrด*?๏?MxฟQ?๙ฦฟบ>ยิืพd'?#Emฟด_v??ไพs>ฟ ท~?|HAพ hpฟกaร>ิ๙r?ุพ,๏}ฟ๖ท๐พ6฿?Un~??=๏>วะพ0?sฟฝdcฟ๎ตพA>	P?์ฺ?X์^?3?E[=LVธพบู3ฟะjฟ7?ฟh@vฟBuUฟฮ$ฟeทึพย๒;พ๚ฝ==ฦ<>ฦ?ใ>ไ`??A;?๑ึT?ดg?คt?Q|?ฑ?แ{?c|?	w?๏o?g?K^?ฮoT??J?f์???5?๖z+?L!?๛?ญพ?mๆ?ภ๎๚>่๊>สน?>์aอ> ?ฟ>c"ณ>{-ง>a๕>q>>Kฦ|>ชk>ูv[>?pL>Pk>>nV1>*#%>0ร>็(>uG>l%๘=}?ๆ=ื=จ#ศ=ฎHบ=๛aญ=?^ก=20=ว=?=฿!r=ิVa=bตQ=)C=ถ5=)=J=ต_=7=ศ?<ํ๋<฿?<Oฬ<฿ พ<)๎ฐ<]?อ๕=ผพษ9>าi?๛ตไพ๓ก>k๔๚พ%n?|vฝพaฟ๑ฟฟฒฆqฟu=ิพQ0?็ฦ>?b[ฟน=ฯ?wz[ฟBdn?$hฟl๗A?=ภพ์พV e?ส๚\ฟY๔ฬฝ๗e{?๖ฏพ?lฟฑ:ก>ถO}?๎า>๑aฟฑNฟิeโ=YUb?ุฟi?ม๖>ศ/๋พ!KoฟMsฟ/ฟ7ร	ฝUน๛>IZ?~Q?ีn?A.6?๔๐ห>'_]=๔๑พ6RฟๅCฟงfhฟงฆ{ฟฅนฟpPwฟQ@eฟฝ6Lฟ.ฟ7@ฟIถูพ#พc-พgHฝ๚=?a+>.]>๘ฒ>฿Nฺ>S๑?>ฺ?๒?5j)?ฬh4?>?F?'N?5T?0Z?j%_?Ewc?<:g?$j?ภXm?๘ัo?๘q?ีs?*tu???v?็x?"y?*z?ุz?์{?ภ!|?0ฆ|?๒}?a|}?า}?(~?ฮ]~?ฮ~?Pฦ~?T๐~?น?>4?O?/g?ฉ{?d?ภ?ช?ต?ฟ?.ศ?ฉฯ?$ึ?ภ??เ?ัไ?v่?๋?Y๎?ท๐?pqN??g?พพXฟ๖?ฟธg!?t	ฬ>2ฟจเ>๑}๓>.
eฟืTTฟมFึพ7ดพ(~*ฟฟkp่พซ]?ลz>qฟo?Dฆ3ฟ8?"?ฟ฿H?๘zฟg?็พBzฟy?uจฝ`๑xฟ๎~>`ฯy?ๆTฝ๙,zฟ(
ฟb?๔ศ?i?2*ดพ@oฟซ;iฟยอพฎ>+I?R?ง^c?ฯ?ฑ?ี=๔3ชพyห.ฟํ#hฟี9ฟQผwฟPXฟ(ฟ๒?พILพวm?<Ez>dน?>ี?ฅ:9?xLS?bf?่เs?E๏{??;?ฎ|?cuw?xp?P,h?b๔^?>!U?๔J?ฺข@?uU6?],,?อ?"?ฅก?^?A?ฐ?>?๋>่พ?>^Xฮ>Bฤภ>ช?ณ>p๚ง>ต>%>A>{~>:ดl>๙\>ตqM>๛Z?>?52>ข๓%>>?><๐>็_๙=m#่=uุ=ฝ!ษ=F5ป=F>ฎ=,ข="๏=?x=ดป=ๅUs=ub==ภR=k!D=เ6=น?)=K==ไ=ศ?<zํ<nค?<ฏSอ<๛ฟ<xฯฑ<ib?,)^?^	?cมU?๎ดF?Eหjฟถv7?;
fฟ2a?#ตไ> ฟฃhฟ
oฟm๗>ฟำ~ฝDd?; ?A8xฟ๗เฉ>>ฑด> a6ฟ0T?BMฟ๘ณ?pRพะYฺพ็ืs?ัAHฟ=mพ/r?Pวnพ}ฝuฟฒั_>jง?ั/Y>Wฟ9โXฟ.ุ'=Z??o?\%ถ>ำพ๔jฟจvฟุาฟ,!ฝRG๋>q๗U?ณ~?ใpq?m	;?]ืื>%=0พ่ฟถต@ฟtfฟษึzฟขเฟ|<xฟภfฟ!Nฟต0ฟฟอU?พตฎพ!ฤ5พ]iฝLe=IU$>"'>ศฏ>Gณื>ู๛>ฬะ??ข(?๛ฑ3?้s=?ฆF?ฺM?pT?าY?aำ^?ฮ/c??f?Jj?ถ)m?ฉo?ิq?อถs?lYu?ซฤv?ว?w?y?	?y?zหz?}{?ๅ|?ฆ|?}?๘u}?๙ฬ}?X~?ฃY~?2~?0ร~?ํ~?`?62?ศM?ฉe?Wz?@?ร?1ฉ?ำด?ๆพ??ว?.ฯ?นี?d??Lเ?ไ?:่?i๋?,๎?๐?bํn?qZฬ>Mพ*b>u}?neพD๚ฝ<พ _?ษ=ฟา~ฟซFฟลJ4ฟ9fฟXtฟ-พึuz??ฺ@ฝ|Qฟ+~?owYฟr1:?%??ฟฏ[c?๛ใฟฬgR?พ q:ฟ๕n?tคn=น)~ฟ?่3>๐%~?ว(=ฮtฟฟถฟ๎>ึไ?(ํ?wไพiมiฟ!onฟ?ใพre>ปB?{d~?Yg?ฝ?ฐ>แ๋พช)ฟฌPeฟตฎ~ฟXyฟ[ฟใ+,ฟช็พx']พo?~<l>๊ื>\C?-7?มปQ?ูze?ล$s?{?Vu?็ฒ?e๖|?แw?ฮ?p?ขฦh?#_?cัU?ฉK?mXA?	7?.?,?ำ๋"?ิG?9??ำ?44?>ปํ>ฟร?>Nฯ>6ฌม>ฬึด>Eวจ>ฐu>ู>๊><>ตฺm>]>~rN>J@>3>ฤ&>ูG>$>?>\๚=XH้=G&ู=ฯส=?!ผ=ฏ=-๙ข=ฎ=*=(a=๊t==c=หS=สE=
m7=?ฒ*=ฺ=fิ=๗	=ฎ =เE๎<?ป?<ุWฮ<ภ<วฐฒ<Wุทพฉนj?}?งy?ไ>๔uuฟ~?}ฟO7๛>วU?ๆ1ุฝ?!"ฟcพ5ฟ๏?พ๔ป>๖ส}?฿S>QทฟT?ะbุ=Mฟwฑ/?pภ)ฟสR๋>O๏ผฝำฟ๊|?ใm/ฟkฐทพญ?S่๔ฝ|ฟใ๕=ฦศ?v>ณKฟฌvbฟุ่๋ผT๏P?Lกt?ภะ>cบพfPeฟeyฟ>.&ฟ๔%ๅฝ?ฺ>YQ?ฝ?}?#ืs?ฒภ??ๅใ>ธฯ=ฌ(lพpฟYt=ฟรodฟV๕yฟ๘ฟ|yฟ85hฟe๛Oฟ$ึ2ฟมสฟ๏โพ๎6?พa|>พฺ'ฝF=ชF>Y฿>๕ตฌ>๐ี>F๙>ฌฦ?ฅ?Jส'?u๚2?ขา<?5wE?
M?2ชS? tY?๙^?่b?ฝf?ชj?s๚l?o??ฐq?ืs?>u?Zญv?๋w??x?ิ๎y?Kพz?!r{??|?|?
}?o}?dว}?~?sU~?~?ภ~?็๊~??+0?L?!d?y??ฤ?Uจ?ด?Aพ?ว?ฒฮ?Nี???๛฿?Fไ??็?5๋??ํ?h๐?ค๘N>JGy?`?*?ษU[?MจQ?3mUฟชพ ?P/ฟปฏ?g<zฝmฟ๚yฟใnฟ	ฟ?๕Hฟ/฿S>y~?hซพ&๒!ฟ*}?sฟ"\?ผ6^ฟGu?ๆ-}ฟh7?ิบบQฟี`?ป9>^?ฟ=m๒?ฯ๋>zmฟc๓*ฟ%ฬ>ึม~?D,!?l#rพcฟz๛rฟฉ๙พญ7>@;?ซ}?Yk?คม?m+>>พMK$ฟMWbฟ์?}ฟRWzฟ๙ภ]ฟย/ฟ2๏พ&ซmพ่v9
ท]>?Vั>ธซ?M5?ึ$P?pUd?=cr?"{?เP?ไศ?;}?Jx?Eq?_i?SB`?<V?p\L?B?ฝ7?g-?^#?ํ??#ฐ?MV?>#๎>Oศ?>Dะ>๛ย>ษฐต>?ฉ>ณ5>r>r>ว;>o>ฐ^>:sO>0:A>๕3>|'>$
>;H>พA>สิ๛=?m๊=7ฺ=?ห=pฝ=ึ๖ฏ=Bฦฃ=?l=Y?==๎ฝu=๐ฒd=๑ีT=(F=2T8=+=ฟข=พ=c?
=ง =Er๏<ำ?<\ฯ<3๗ภ<ณ<bทzฟ๓i>g/????ๆฟb^ฟVE]?.:ฟปฉJฝ?|พ>lพูธพด์iฝ???uz?`?ฝH:qฟ,DF??พmh?พqญ?w8?พkฌ>\>~2ฟ๏??Lฟป๕พ๗ภ{?๎ปacฟะถฆ<ณ}?1ม>>ฟBลjฟษฝ+ๅF?mฝx?ษฒ๊>ภ0กพช_ฟู{ฟ >.ฟ4hพxษ>uํK?ฺf|?Uv?'SD?6๏>wฤ?=Vพ~า?พF!:ฟXbฟ^yฟ  ฟcํyฟูiฟwัQฟ๐4ฟ	ฟข็พปคพ๙0GพฒฝJ๐'=6>ชmy>หฉ>฿vา>๎๖>ฑป?๚.?/๙&?:B2?ถ0<?/้D?ฎL?|=S?Y?3.^?ไb?ฟ~f??i?๗สl?ฦVo?q?บxs?#u?ํv?<ืw?่ํx?฿y?ฑz?ฆf{?|?s|?ฅ}?i}?ษม}?ง~?>Q~?์~?ใผ~?*่~?ฆ?.?<J?b?ฎw?๒?ฤ?wง?Tณ?ฝ?ฦ?5ฮ?โิ?ฉฺ?ช฿? ไ?ภ็? ๋?ัํ?A๐?7ฟ#ฆC?๏?ฺ w?0U>z_~ฟHWl?ทBzฟJQ?ฐKไ>ฟn%ฟอyฟ?ฟๅพqฟVฟl?ญSi??ฟฬพik?U&ฟzfs?ูvsฟ,ึ~?{rฟ|?>ndฟMN?`y>l~ฟใฝ=0??f>Oีcฟ7ผ9ฟลdจ>aa|?qฝ-??5พ}\ฟ?vฟิฟ:ฺ	>
4?]{?[?n?z!?_๋J>z๎}พf?ฟM8_ฟ*}ฟv{ฟcU`ฟใH3ฟ5๗พ+~พ?๔yผไ]O>ฮห>K?R?2?ฤN?.*c?Vq?Tซz?,(?3??|}?dฐx?zr?๕i?๐ๆ`?ว-W?ฤM?๏ภB?ลo8?	=.?mB$??~<?1H?๙w?>
6๏>ฬ฿>\:ั>{ร>?ถ>`ช>๕>DA>ฦ:>8ู>l'p>๔ย_>้sP>บ)B>wิ4>฿d(>jฬ>N?>y๊>3?= ๋=฿G?=๊ฬ=๛ฝ=ำฐ=Wค=๊+==ฌ=๐๑v=กัe=ษเU=
G=[;9=%a,=๙j =I=ฯ์
=xH=ช๐<๋฿<*`ะ<O้ม<dsด<ฐ 3ฟ0%ฟ๔นผศพ๖czฟๆ=oรฤ>Wพ5
ฟy$e?ั]C?ต๒_> ]=ฤvจ>๎ๅ[?{Z?]ฉาพ Nฟฌj?ว,ษพVวฆฝแฅ>CAพนร=ี+ค>INฟฟ}?r"็พPฟt?<	ใ=้ฯฟ9๒ขฝ7ly?iz้>a-0ฟยqฟ_+พ]<?๑{?๓?พE)Yฟoซ}ฟr?5ฟ Bพf&ธ>IF?[์z?? x?ํฟH?ซ๚>2๐>ฒ๗?พRฉ๖พอผ6ฟ่.`ฟ๐?wฟ_๘ฟ'ฒzฟื?jฟBSฟ?7ฟ๑Cฟธ์พ<ฉพฟแOพฆฝ>	=ก#>J๙r>|฿ฆ>ึฯ>็๔>?ฏ
?A?Q'&?K1?%;?ZD?ฮL?OะR?ยตX??]?rWb?ญ?f?-ฆi?Cl?R-o?iq?xYs?lu?c~v?ัยw?3?x?3ะy?พฃz?[{?๚{?ษ|?#๛|?b}?&ผ}?ล	~?M~?C~?ทน~?kๅ~?E?,?sH?a?Wv?ษ?ร?ฆ?ฒ?๔ผ?๐ล?ธอ?uิ?Kฺ?Y฿?นใ?็?ฬ๊?ฃํ?๐?"yฟmxฝe!?แ๕>ฯงฟีd0ฟ๗นw?7lฟ๋ศล>@bU?k[พHฟ>cฟL(?ฟ/พฦF?S =?ฯ3MฟzพรsJ?่|ฟb?~?ใ{~ฟ๛ำ~?#`ฟo็>>ฟrฟxQ8?ศดี>)uyฟ>พพOแ{?@+ก>
โXฟqPGฟฒ>lฦx? 9?7ิ๐ฝr้Tฟzฟิ๘ฟฦ}ท=5,?tDy?'r??(?_j>.ข`พ Mฟ2๔[ฟว1|ฟu|ฟัbฟiพ6ฟ#$?พฐ?พ4๛ผุ๙@>พอฤ>1k?:฿0?ไL?๙a?ะp?ถ5z?9๛~?ำ้?๊บ}?y?kr?Dj?๚a?ฺW?ภM??sC?ย!9?์.? ํ$?8?%??๛฿?L ?ฎH๐>ะเ>่/า>๚bฤ>Qdท>-ซ>oต>๕>
ใ>v>จMq>ะี`>tQ>9C>ใณ5>:5)>ซ>[ฒ>1>I?=?ถ์=ฅX?=๒อ=็พ=]ฏฑ=i`ฅ=ี๊=ะ?=Q=๑%x=R๐f=ก๋V=แH=":=G8-=23!=n=:=?้=ห๑<จแ<Sdั<j?ย<ฒTต<$e>YฟทFฟR`ฟ*[ฟ9?8พ+ฌร>ฑ lฟ้n?๕z?ขณ?ทw๊>ZG*?5|?6!?8ซ,ฟโฟ๐}?zฎฟ_ด>?ภ=?ฝ฿ฝรฝs^๗>pdฟt?Kฃพมจ1ฟ7ขh?f>์H}ฟญ๘6พc?r?ญ?ง ฟ๎bwฟdqพฉX0?O9~?$?[พNRฟn๘~ฟl=ฟwLiพ?ฆ>m๒@?ก1y?ญรy?*M?e๛?I่/>๏น)พeํพ>G3ฟ+๓]ฟ!่vฟ5แฟฝi{ฟ#UlฟทdUฟใ9ฟ?yฟร๐พไนญพXพฅถฝฒี<d>Jl>ๅ๑ฃ>ง3อ>x9๒>1ฃ	?5S?ฒT%?ฉฯ0?๐๊:?dหC?iK?ชbR?VX?]?ฌb?Q f?oi?Vkl?ฌo?ุDq?:s?)ํt?ฝfv?Mฎw?iสx?ฦภy?^z?O{?๐{?{|?๓|?\}?}ถ}??~?รH~?~?ถ~?จโ~?แ??)?จF?}_??t??ภ?นฅ?าฑ?Lผ?_ล?:อ?ิ?ํู?฿?rใ?F็?๊?uํ?๑๏?Q6ญพ3Vฟ์/;พNถพf๚uฟz$ฝ2D?0ฟํ(พ๖?ิ6>q๎?พHจฟ*฿พ$:D>=q?t?๛>ฐฦpฟuv>๚j?lฟ๚๕}?Mฯ~ฟวu?ซนFฟญ>ใ>Pศ{ฟ?fำ?+qฟ4Fพv?*กอ>ฌNLฟSฟ๐<>v๕s??D?5 kฝLฟN|ฟ<	ฟMอ5=$?ษv?S๓t?โด.?ซ๙>$Cพ,กฟXฟฎ{ฟvU}ฟL5eฟฺ":ฟAฟฯfพ]<ฝฆ2>ธ{พ>ย
?น.?\;K?Aย`?{?o?Kปy?
ส~?ฤ๔?๖}?๛ty?s?k?n+b?๒X?/pN?ๆ%D?ำ9?/?%????zy?w	?? ?๙Z๑>Sิแ>9%ำ>3Jล>?=ธ>m๙ซ>(u?>ฌจ>=>๔>ฯsr>่a>uR>ญD>F6>*>ๅP>eg>ๅ;	>๓?=ี?ํ=hi?=๘ฮ=ิฟ=ฒ=z-ฆ=ฟฉ=๑=๑๖=๐Yy=h=w๖W==๛H=ฉ	;=j.=k๛!=ลฝ=ฆG=@=t๗๒<6โ<|hา<อร<6ถ<ล็p?ุ1ฟบฏ{ฟุ0tฟ?พผ?77IฟX์V?~|ฟD้}=ฐขv?น-g?"H?Fpf?vA{?ฝ'ฌ>%โ^ฟ๏ํญพY}?๏งJฟ3๕ร>J? พ3Bล=ุพใa!?[ZtฟSe?๔9พHฟtY?ปซ>ใีwฟRพHrj?6A?ภฟ?{ฟฒ-พฆ๎#?๏?qโ?u&พ:๓JฟjฟฟqDฟ4พtฬ>;?7w??N{?%Q?A?วG>gพไพ๎ภ/ฟwฅ[ฟมuฟบฟ|ฟฌmฟม!Wฟ;ฟ=ซฟฅ๕พO3ฒพ&7aพ4๓ฦฝ5ซ<u๙ >ผf>ฦก>ส>W?๏>ฐ?d?R$?T0?G:?ก;C?~K?๔Q?ู๕W?ฉ3]?ลa?ฌภe?ฎ7i?0;l?ีูn? q?s?ฦัt?๚Nv?ฐw?ธx?Fฑy?๎z?เC{?่ๅ{?Ur|?์|?sU}?ฬฐ}?๐?}?~D~?โ~?Sณ~?แ฿~?y?็'?ฺD?ํ]?คs?s?ฝ?ุค?ฑ?ฃป?อฤ?ผฬ?ำ?ู?ต??+ใ?็?a๊?Gํ?ษ๏?เ๋?icpฟ?^ฟลbeฟมVdฟึ๔?|<ฒA๕<Y*ฟธ>e?f0?<คพnฝh?ฏ?? I>HฆฟใMฬ>ิศ>A฿Nฟlพp?mtฟQc?L'ฟ->5?ฯฟ๚?น ?ฉซeฟqฒดพลm?O๘>V3>ฟ?^ฟย(แ=~๔m?ลูN??;D;(Cฟm~ฟ๏บ%ฟ pบ_ฃ?_ํs?โw?๖4?>ฝz%พูฟ??Tฟpำyฟี~ฟEgฟๅu=ฟฬaฟแพ/{ฝ$>!ธ>]?,?I?ฅ_?'o?<y?~???.~?<ำy?~s??ญk?Mหb?.Y?DO?ืD?:?[H0?ฏ@&?ิ?{?ฦ
?8m?๋l๒>ลืโ>Qิ><1ฦ>Aน>ชลฌ>ว4ก>B\>a3>>ฑ>แs>V๛b>ขuS>๘D>?r7>?ี*>>i>ไ	>%_ >จ ๏=&z?=๚ฯ=คภภ=?gณ=๚ฆ=งh=Bฃ=a=๎z=ฏ-i=MY=๓I=ะ๐;=ๆ.=ฃร"=x=๕=ค,=ุ#๔<ร1ใ<คlำ<?ฟฤ<Oท<ษ	K?เฐ>ุ?พ๊Pใพo~็>ภN?j๖ฟโ??ฟ2โใพm?9?@??ฃ๎w?อ?h+Y?ุAK;ฆ?zฟ~@Vฝdผj?kฟ4ฬ?ฉฎพ92>ฃ{๋พ?A?
}ฟA)Q?uฝฌ๔[ฟไAG?ร)โ>ไoฟiพฝพัเ_?ฯX+?-?พบr~ฟSาผพบำ?ต??ิ#%?ไรโฝ4Cฟ๙?ฟbAKฟ`พBะ>R5?Q?t?ข|?วU?i?_>n๚ฝฺพ/*,ฟ๚EYฟฉtฟIฟ:ฑ|ฟd฿nฟQึXฟ้=ฟุฟA๙พวจถพs?iพu^ืฝ}4<ีร๓=ณ_>&>ส้ว>}ํ>Z?=t?2ญ#?NZ/?ข9?IซB?J?๙Q?FW?i฿\?'|a?พe? i?า
l?ฬฏn??๛p?ฯ๚r?Bถt?7v?๛w?ฆx?ณกy?n{z?,8{?ฤ?{?i|?fไ|?ฺN}?ซ}??๚}?4@~?*|~?ฐ~??~??ฯ%?
C?\\?Ir?F?ธ?๖ฃ?Kฐ?๚บ?:ฤ?=ฬ?-ำ?/ู?b??ไโ?หๆ?,๊?ํ?ก๏?p?Tพ๋pฟWุpฟีfฐพเะx?ชJฟWฒ?ต vฟ??ภr?M]ไ>จ7->์ง>*L?ฝq?sl๑ฝ?xฟtำ!?ซ>8จ%ฟิูW?mฦ_ฟุะH?1ฟ9คๆผ๑1?7~ฟ$ษ>๓&8?Wฟ	็พมc?๏k?ซ.ฟฤ๚gฟP= หf?ล3X?ฮ=่:ฟlฟ
/ฟUL=ฝ_๎?ฑp?๕นy?ว;?๕ฃ>ตฌพ๕ฟสNQฟ<nxฟถ~ฟ?ฒiฟ9ท@ฟa9ฟSพ?=ฝ?>๓ฝฑ>ฺ`?[*?์ึG?PC^?XKn?ธx?๗Z~???๗b~?ห.z?ก๗s?ะ<l?ic??ึY?CอO?JE?U3;?๕0?ษ้&?ข$?)ต?ลฅ
?3??~๓>๏ฺใ>.ี>ว>๐น>ศญ>N๔ก>ฤ>t?>{N>?ฟt>?d>vT>t็E>๒Q8>!ฆ+>Hี>iั>B
>M? >v%๐=เ฿=๙ะ=+ญม=Dด=วง='=๚T=ะA=๋ม{=\Lj="Z=๒๋J=๖ื<=ญฝ/=?#=r2={ข=ฮ=<P๕<PIไ<ฬpิ<ปฑล<๘ท<ฌฌฝซ^u?์#ญ>ญ>xSp?ํโp>ุNฟป6N?NฏพญAUฟภฃ> e?O|?ืq?q?4)ฉพ7~ฟVu>;]F?ท6}ฟ-C?}ฃ	ฟฃฎ๘>ฦฟเz\?ๆฟฆ8?ฯMึ=ใjkฟำิ1?ส
?{tdฟ]E์พf_S?ซ';?ุพvืฟง?พ
	?;|?฿/?@๛oฝภะ:ฟ๚นฟฯขQฟeฎพัNa>า.?ฮr?7พ}?้X?๗e?ๆ*w>ซอฝhัพX(ฟ็ิVฟ+?sฟ>ฟA}ฟ<pฟTZฟฦ?ฟ!  ฟy?พ2ปพB{rพ!ฦ็ฝf;ตๅ=@Y> >kBล>๋>2x??Tุ"?.??8?^B?J?๎Q?H4W?ห\?h2a?@e?.ศh?;ฺk?n?Oืp?๕ฺr?t?v?,pw?x?y??mz?j,{?ั{?ถ`|?ฟ?|?8H}?Vฅ}?๖}?ไ;~?nx~??ฌ~?Jฺ~?ข?ต#?8A?ศZ?๋p??ฒ?ฃ?ฏ?Pบ?ฆร?ฝห?พา?ฯุ???โ?ๆ?๖้?๊์?y๏?อx๏>*?ฺLดพ_ะพฦ>Ye?_uฟซ?q?9ไuฟ
ห=ฌ|?n๘J?ฅศ?*??u?FG?zืพ๖\ฟตpQ?v	?ฝWๆพC}4??ทAฟ๖ '?aฒพ่EพฺL?ห(xฟม>KฟL??ฏEฟZฟ/V?)p#?พำฟ๑oฟ(
#ฝ้^?tค`??=ุ/ฟด?ฟ้๊7ฟU:ปฝ~๖	?ุm?ส{?ณื@?z6ณ>3ำฝ๘ฟ์{MฟDๅvฟz7ฟ หkฟๆCฟฑฟงพชผฝฬ>ศRซ>จ?8#(?ัF?J๛\?ุim?N/x?~?{??ค~?งz?znt?ืษl?Ed?ิ}Z?*zP?ฃ6F?bโ;?:ข1?f'?
ศ?R?<?๗?ย๔>ฯ?ไ>ะึ>ม?ว>ษบ>ว]ฎ>ผณข>1ร>w>ซ๋>ลๅu> e>vU>วึF>:19>_v,>q >d>๊5>r>?J๑=เ=๕ั=ฏย=T ต=ขจ=sๆ=ฐ=>็=ๆ๕|=kk=๖[=LไK=ฟ==ฮ0=T$=ศ์=ๆO=lo=?|๖<?`ๅ<๔tี<ึฃฦ<ู๋ธ<หEbฟํM?o?ืi?ป๙k?๗wใพ?พ*ง>t>H~ฟ$พw?{T?Y??jณ>H ฟ9hฟ\ษ?ม5?~ฟ`กd?>5ฟ์['?nBฟ	p?'7{ฟ๖?V}{>๚ฒvฟซ?ง"?YฟVฟ?7ฟผ	E?wI?J{ฒพฬฟ้0?พำ๕>?	~?๕:?&4ฮป๊2ฟํ~ฟะคWฟ๊%มพ6ฐ<>-^(?7ฮo?ก~?ษ\?ฏ?T>W?ฝ4`วพมฬ$ฟpRTฟ?ไqฟR้~ฟร}ฟ'>qฟบ%\ฟAฟ#"ฟ=ฟwฟพi{พ๒)๘ฝCwปน\ื=uR>},>tย>์บ่>9h??ท"?0โ-?ยW8?เA?กI?mงP?แาV?ฯ5\?V่`? e?h?lฉk?'[n?vฒp?๖บr?ุ~t?v?D[w?ex?Uy?<`z? {?Vว{?ึW|?ี|?A}?}?๑}?7~?ญt~?ฉ~?zื~?2?~?!?d??2Y?o?็?ช?/ข?มฎ?คน?ร?=ห?Oา?oุ?ผ??Tโ?Nๆ?ภ้?ป์?P๏??j๛พ๕??3๘>ม>vvi?Oอ>Noฟ๘v?ึ*ฟจxใพL?[ษz??Y?APf?่?Lฦ?B.ฟญ,ฟWrq?ฺPตพ๒hพร_?ฟ29?>น4พ๑ณพ>\a?ฃlฟถแ>1^?ก1ฟ*!ฟG?S
5?อฟศXvฟ๊ฝ#U?ถ"h?ฦฃ;>?%ฟบฟ(^@ฟถพัฟ ?1i?ผ+}?sF?Iย>zฝ\ร๗พๆIฟย8uฟฟLสmฟGฟmฦฟฏพ??ฝC๙๐=ะ฿ค>Qิ?>ฎๅ%?ฺZD?ญ[?l?ลกw?๛ฺ}?ฎ๛?&ร~?ฮ?z?ใt?๔Tm?[กd?y#[?๘%Q?ๅF?ฒ<?BN2?:(?	k?๏?๘า??ง?๕>gเๅ>7๘ึ><ๅศ>ขป>ฆ)ฏ>sฃ>v>i+>อ>w>3f>฿vV>ฦG>y:>F->Y!>Z;>?>6>o๒=Iฌแ=๎า=0ร=?ต=ฌaฉ=Xฅ=fธ=ฌ=เ)~=ณl=ษ!\=ฅ?L=@ฆ>=๎k1=J%=ง=P?=ฯ=ฉ๗<jxๆ<yึ<๑ว<9ปน<_ฟ๛บ่_?ฦm??า>Fjฟ3{ด>!พ\\??๋XeฟEฟM>๊l?R฿>8ฉฝฆYฟ4;ฟ?<?ธ+ช>๕hoฟดIy?ํฆXฟ&XK?6^ฟH๛{?0ซoฟใ๒>๙Uร>จ}ฟบJ?>?X8?๛Fฟ?ๆ ฟ?4?tV?๐EพP~ฟTอฟ?ื>Vช{?วฃC?G<=?่(ฟ)}ฟชD]ฟำพ3ั> ธ!?>ฺl?L?ข`?ฏ??8 >?eฝzฆฝพม!ฟษพQฟ ypฟ~ฟร8~ฟ]rฟrภ]ฟ?๘BฟB$ฟฅqฟ}๐รพ`ึพาDพv<ผ%ษ=e
L>7>้๎ฟ>*Wๆ>pW???^,!?%-?bฑ7?ะ๖@?คI?u7P?qV?vเ[?๑`?;ฟd?ฌWh?exk?0n?qp?ัr?๑bt?อ๎u?CFw?.px?ry?Rz?ป{?ฝ{?์N|?Uอ|??:}?ฤ}?๛๋}?53~?่p~?Zฆ~?ฆิ~?ฟ?~?y?=?W?+n?ต?ข?Iก?๛ญ?๙ธ?}ย?ผส?฿ั?ุ?h??โ?ๆ?้?์?'๏?ญฟฐ?๖z?F็m?>lr?ตพฯ<ฟQี#?X'พ!Uฟช๗ื>r๚w?=ฅ}?|?ฟf?Z>,`ฟ฿?พพu?ฒฟ4Y๚;Jซ>o?พ/ฅ>ฑาื9?8 ฟ=Rq?`C\ฟ๘#q;์8l?4ฟ~๒5ฟฉ6?pE?yv๑พ%'{ฟ0/AพปJ?tฆn?จx>=ฺฟ1ร~ฟค\Hฟ}9พ๎>งศd?Cc~?฿ิK?ไ*ั>?ข6ฝ๊f๋พ`pEฟ๓hsฟ฿ูฟํฏoฟ๐JฟG{ฟLทพศ_๛ฝBฮำ=We>cN๚>yข#?B?OZZ?k?zw?จ}?2๔?z๎~?@1{?QUu?%?m?ุ:e?สว[?ญะQ?G?C>=?ญ๙2?"โ(?ก?9?*i?ุซ?1ฑ๖>ตโๆ>b์ื>หษ>S{ผ>f๕ฏ>K2ค>ฮ)>Jำ>โ%>S1x>Eg>,wW>JตH>ฏ๏:>ฦ.>ฑ">L๐>0>ณำ>ย๓=๗ผโ=ใำ=ฏrฤ=ยุถ=ด.ช=:d=j=2=ุ]=]จm=,]=?ิM=d?=C2=ไ%=ta=บช=2ฒ=fี๘<๖็<C}ื<ศ<บ<6hkฝฝzMฟ*J>้ฝ>คพ)จuฟ๊ผX?งตDฟ๏|?ัฟjhฟถT~พ'
>js=<ป?พ zฟว=๗พ5๔f?่๕=PำPฟ??ฐ?qฟๅศf?Orฟ???ๅ]ฟ{แช>v??ฟUล>	K?/4ฟต๙3ฟud#?:พa?\RFพ.g{ฟQฟmGน>{_x?CL?z!ษ=Xฟuร{ฟฬbฟๅบๅพบ|ๅ=โ?คฉi?าพ?zVc?๔"?ด>แฒฝGุณพด1ฟ'Oฟฤ?nฟv~ฟ?~ฟ?psฟlR_ฟ๔฿Dฟี[&ฟTขฟ+UศพพxrพJxผเ๊บ= E>$A>ะBฝ>ห๑ใ>ฺE?ุฌ?IU ?Sg,?a
7?.d@?#H?วO?ึV?ฟ[?9S`?)~d?
h?%Gk?ผn?Ahp?zr?้Ft?{ึu?)1w?แ]x?ซby?ศDz?ฮ{?ถฒ{?๗E|?ล|?#4}?๐}?๏ๆ}?ึ.~?m~?ฃ~?ฮั~?H๚~?X?ต;?V?ศl???c??3ญ?Lธ?่ม?:ส?oั?ญื???ยแ?ะๅ?S้?]์??๎?#xฟข%พI?ฟ๓h?#จ๒>์
Uฟๅkด=อiธ=6ฦ>zฟH๔RฝาC?u?๏q?๑/?าภ?ฝ{ฟฯพoz?๐Cฟ(x>๖=๐uพ๔B>n5>f๘"ฟ"{?ว_GฟทiพIv?าถฟv6Hฟ!.#?๙gS?~ษพ6T~ฟ?พ?V??จ(t?ุั>๙ฟC}ฟแOฟl๓fพ็M?>๏`?๔C?แ๙P?ฐึ฿>คxผH??พ9Aฟvqฟ?๛ฟฎ{qฟrMฟ๒#ฟกjฟพ8Rพถ=จใ>พ๔>ฏY!?ว@?jY?ืฅj?qxv?J}?้????{?Lลu?ien?ธาe?ลj\?GzR?A?H?๋=?|ค3?@)?ัฏ?(???๕:?aม๗>บไ็>Rเุ>กฑส>๖Sฝ>มฐ>l๑ค>??>{>๊ย>๚Vy>๖Wh>kwX>{คI>?ฮ;>๎ๆ.>ศ?">8ฅ>อ/>ฮp>|ธ๔=ขอใ=ีิ=+_ล=๖ดท=ป๛ช=#=อ=ื=่H=วn=l7^=TอN=t@=.3=ธฌ&=ษ=$X=S=ศ๚<ง่<jุ<%zษ<ิ}ป<oO?&3uฟjฟ_Pิพ:laฟR๒ฟ1?ล๕~ฟ๗๘k?kกฝฉฟaป%ฟ/mพMaงพX๕9ฟ๛?}ฟ3?พน}?ัUTพ?ภ$ฟu^x?#%~ฟ๕x?L}ฟผ๑{?ีlEฟIจ=> ?วู}ฟuV>๙\?ฟn@Eฟb_?สVk?จ้ฝwฟเ*ฟN๊>,t?%๏T?ํบ>fgฟmgyฟึSgฟ!๗พบ=@??9=f?t๘?ฎyf?r๎'?[ช>็XEผช๖ฉพ๕MฟมdLฟจomฟใ}ฟ๛~ฟอytฟ?`ฟTภFฟฆp(ฟฯฟhตฬพ&eพศพโุผ^ฎฌ=บ?>>fI>0บ>ำแ>x3?น?z}?฿จ+?มb6?๚ะ??H?#VO?3ฌU?ช4[?.`?อ<d?)ๆg?ฌk?ผฺm?ๆBp?Zr?ม*t?พu?๗w?~Kx?นRy?๖6z?ำ?z?Sจ{?๗<|?ลฝ|?b-}?}??แ}?q*~?Pi~?ฦ~?๓ฮ~?ฯ๗~?3?ฺ9?fT?dk?N??|?kฌ?ท?Rม?ธษ??ะ?Kื?ภ??yแ?ๅ?้?-์?ี๎?G๕น>pฟ๚=hyฉ>Nถพ;s~ฟ|),?ส๛พ.สQ?seฟจIฟาธอ>ศB????h?ร>?ๅพฌใ}ฟno>ฐฟb?ข$gฟิLํ>ฺ็ฬฝk๓ฝ!ฮ[ฝzฒ>ํCAฟิ???e.ฟพญ8}?จ๖ะพ+Xฟ3ฌ?0่_?๊?พตฺฟฟขชพ๗3?^ฃx?ท>ฑ๊ฟ๑ฝzฟ7่Vฟ๛๓พ๊ว>แ[?อ?แU?0I๎>`zi<?(าพแ<ฟ~`oฟต?ฟ^-sฟส้Oฟ$ภฟ?วพ๘๋พcX=[>&%๏>c?Y๕>?๘ขW?hฏi?ญ?u?]๛|?-ฺ?;?ะ{?๚2v?ฟ๊n??hf?j]?ว"S?๚๊H?)>?ฎN4??/*?Q ?ฤ?ภ?ูษ?5ั๘>tๆ่>ิู>ห>r,พ>ฑ>tฐฅ>>?">ไ_>|z>Hji>wY>?J>?ญ<>ท/>ู#> Z>fุ>็>0?๕=H?ไ=ฤ	ี=คKฦ=(ธ=ภศซ=?แ=อ=๏|=โโ=ญๅo=<B_=ซลO=ช[A=L๑3=๎t'=ึ==๘๔=+.๛<ฟ้<ู<?lส<!_ผ<คn?C๒ฎพญัฟฏqฟ๘ฅwฟ~แ=v=?็^ฟ#ถ?ใ>ล[ฟmjฟ8j&ฟู)ฟlฟฑะdฟุ8>งg}??ฉํพฃ?พุb?(ท~ฟ{ำ?กฟ?๑o?็แ'ฟB=0h;?2wฟFv>Rตi?ฏ#	ฟTฟ๗4๘>ฺ+s??(
ฝU]qฟQ 7ฟ,ฮs>Wo?ฒ\?^yN>ฟMvฟพkฟ็rฟฟ =วฎ?฿b?X๙?ฃpi??ฬ,?}ต>O2$<ด?พโ[ฟฯIฟๆัkฟ๊๙|ฟHฟ~wuฟ้[bฟ๊Hฟ|*ฟa๙	ฟัพจพ?ฦพ่$ฝตo=Ct8>RP>ๆท>F"฿>K ?ฤ?๐ค?ฟ้*?บ5?5=??G?สไN?&IU?8?Z?ัผ_?(๛c?ญg??ใj?ฏm?`p?9r?wt?ฅu?ซw?9x?ตBy?)z?ษ๐z?ๅ{?์3|?๐ต|?&}?4}?ฤ?}?&~?}e~?w~?ฬ~?S๕~???7?ษR??i?~???กซ?๐ถ?ปภ?5ษ?ะ?้ึ?k??0แ?Qๅ?ไ่??๋?ฌ๎?a๑z?ฐ฿Uฟ?<ฟ].็พ%?Xฟaๅ0ฟ?ฏ{?๑?fฟปด?ฟ4?ZฟUx"ฝKฟฺ>gเ>ถBเ<t8ฟ,gฟพุ>๙):?๋j{ฟท(?๋}กพ1ธ+>฿)xพฆF?ฏGZฟL~?ฒุฟฌลพ๎?ทพXกeฟซz๑>cxj?ฤjพธฟdRฮพ?ั%?ร|?Nำ>?๊พโฒwฟzl]ฟพ%?พณ>yตU?ม??Z?ฑ~?>Eู2=Lลพณj8ฟn(mฟC?ฟอฤtฟฒบRฟO!ฟ~ฯพ1|,พ-!x=ุห>G้>ฉท?|=? ?V?หณh?1<u?hจ|?ฃว?b]?N|?\v?&no?ค?f?ทฌ]?*สS?ศI?{B??C๘4??ี*?๕๒ ?N`?"*?X?ญเ๙>ๅ็้>}วฺ>E}ฬ>วฟ>้Wฒ>boฆ>C>ส>ะ?>ข{>|j>ภwZ>บK>=>(0>ไa$>>๛>?ช>เ๗=๊๎ๅ=ฏึ=8ว=Xmน=รฌ=ฺ??=0=X"=?|=Sq=M`= พP=ฬBB=kศ4=#=(=s=๖ฒ=Z=Z?<ึ๊<ธฺ<Y^ห<o@ฝ<ๅJ>๐ฐ?ฤ-ฟkdฟ	ฟ9??,;>ห฿พ\AD=g U?เษฟmฬฟugฟ)0fฟp็ฟ๋v1ฟภD?>Y๗g?cน/ฟHแ@พF@?ช๎rฟ๑_|?X^xฟp_\?พภฟ"๘ฝ]dR?น&lฟ๎yฟ<ธ0t?ฆSโพฝaฟRอ>a.y?#ั?=XKjฟท	Cฟ?ภ2>ผi?นc?S>K{ ฟ'sฟพoฟ๑ฟ@n3;ธU?ด^?ม?ส:l?1?สภ>Cj=x?พุ[ฟศFฟ#jฟW|ฟคฟjvฟLำcฟงlJฟG,ฟขฟ-hีพL้พ?์$พึ)ฝ/=ฯ่1>๎U>k5ต>)ธ?>V ?0ฯ?ญห?๑)*?ฃ5?฿จ>?G?๛rN?ฑๅT?iZ?"q_?;นc?ฅsg?ฒj?)m?ฏ๗o?ลr?๒s?ืu?G๑v?x&x?2y? z?ฑไz?i{?ื*|?ฎ|?ฦ}?K}?ฆื}?!~?ฆa~?#~?3ษ~?ิ๒~?ใ?6?*Q?h?เ|?r?ช?ืช?Aถ?#ภ?ฒศ?ะ?ึ???ๆเ?ๅ?ญ่?อ๋?๎?ไ02?๑nฝ~ฟนิtฟ,?{ฟ3ฝมd?!}ฟ?^?ฦwฝeฟj=๒พผทก<'ซw=CฎฉพUiฟฒc9ฟj๋*?ฑด?)tฟ๎&Q?h๖ฟ๛๚บ>จ+ุพ+4'?ตTmฟ4Uv?>ไพP[ฟแณ~?บึ@พaqpฟD3ร>,s?ใฎพ์}ฟ'ู๐พั?!p~?ศX๏>อญะพ7๚sฟYjcฟTถพ->ิP?ฺ?๐^?ศ9?&=|Iธพ.ี3ฟ9ฮjฟ๋ฟะAvฟ็wUฟํั$ฟ๎พึพ๑<พ==P6>3ึใ>^?@;?ีT?ณg? t??Q|?kฑ??{?โc|?ow?๏o?ฌg?ฌK^?ppT?ซ?J?ํ??9ก5?{+?๊!?ช๛??ฟ?๘ๆ?ษ๏๚>
้๊>นบ?>อbอ>๔?ฟ>*#ณ>6.ง>๖>*r>ฏ>jว|>ทk>ิw[>ศqL>*l>>:W1>่#%>แร>)>H>&๘=?ๆ=ื=$ศ=Iบ=ฤbญ=ท_ก=เ0=มว=ี=๘"r=ูWa=UถQ=๎)C=5=Y)=วJ=_`=ฝ7=๎?<%๎๋<??<sPฬ<ผ!พ<฿อ7ฟe?4Y?=bพY}<>nu?1ูๅพ>๙>๊๛พฤv?๑nฝ๊aฟ;๓ฟฟhqฟNขำพ0?ฉ>?p[ฟบท=?๕][ฟxQn?QhฟOฺA?!๐ฟพ?_พde?ๆ้\ฟ์อฝlk{?รฏพ/ชlฟdก>ูR}?#>9่aฟ NฟYไแ=CNb?ลi?ล>๋พGoฟVPsฟ7ฟงU
ฝ~ช๛>(Z?๔P?ืn?พ26??๛ห>ท]=่พ7Nฟ0โCฟ๊dhฟ๑ฅ{ฟะนฟMQwฟณAeฟ|8Lฟ๙.ฟMBฟบูพ.'พ[-พจHฝฃ์=o[+>AZ>Sฒ>Lฺ>4๏?>ู?ณ๑?xi)?&h4?๙>? F?ท N?ำT?>0Z? %_?wc?:g?๓j?Xm?ำัo?ไ๗q?ีs?tu?ษ?v?ิx?t"y?z?ุz?โ{?ท!|?(ฆ|?์}?[|}?า}?#~?ส]~?ห~?Nฦ~?R๐~?ท?<4?O?-g?ง{?c?ฟ?ช?ต?ฟ?.ศ?ฉฯ?$ึ?ภ??เ?ัไ?u่?๋?Y๎?:iพ
D?ฟ_ฟVฟรe?ไeแ>9:ฟGษ๒>oฅโ>2ุhฟฯขOฟญMศพ}ึฆพ%ฟฟU๒พฉูZ??>ก?rฟI๚m?mว0ฟ5?ิ1ฟช?F?มๅyฟฅบh?Xื?พฟ?y?[[ฝK{xฟ๘>ฃoy?๗mฝ7{zฟฎ	ฟ4	?๋ป?I?,ถพoฟ0?hฟ?หพy>1 J?#^?c??ั=z#ซพฤ!/ฟ7RhฟตAฟ=คwฟ(!Xฟ๒F(ฟOh?พH|Kพ๐?=5{>$!?>< ?
]9?ฅfS?ญf?ํs?ๅ๕{??i?ผฉ|?3nw?"op?"h?H้^?U?ฃ่J??@?I6?ฒ ,?t4"?ฏ?T?2u??๛>ๅ้๋>ทญ?>$Hฮ>๙ดภ>K๎ณ>๑์ง>์จ>ท>6>น์}>ิ?l>ฺw\>ส`M>2K?>D'2>็ๅ%>บx>า>ๅ>0K๙=!่=|ุ=ษ=ฑ%ป=ฤ/ฎ=ข=โ=)m=ฮฐ=As=ฆbb=ฉฎR=D=ฆv6=อ)==ศ=ู=Pณ?<ฐํ<?<Bอ<	ฟ<@yฟๆค$?ค๓U?mฉ๙>๛M?๛nN?ว?eฟYฉ/?Ncaฟ,e?!ษิ>พฟชkฟrฟOCฟ_ฐฝภpa?^ั?๕vฟล#ก>ตผ>)9ฟG?V?qOOฟ?7!?@^พJJีพI
s?แภIฟ๊EdพวL?|Xvพ่9uฟ\ef>N?ตyS>@Xฟ5XฟXง:=LชZ?DDo?P_ด>ไฎิพั่jฟ็svฟ]Cฟ}ูฝ(_์>ืGV?ะง~?Fq?ฃธ:?สื>๔=รพa3	ฟ๙๋@ฟ์fฟๅzฟ?ฟW-xฟงfฟW?Mฟ0ฟSaฟ?พ*bพ๙05พ์1gฝ,Qg=6ฬ$>S]>ษฯฏ>N฿ื>1ฤ๛>Kโ??Tจ(?พ3?~=?๑F??M?T?ตุY?หุ^?4c?" g?Mj?ั,m?ฬซo??ึq?ึธs?0[u?3ฦv?x?8y??y?Wฬz?N~{?|?7|?
}?ev}?Wอ}?ฉ~?๊Y~?o~?eร~?อํ~??X2?ๅM?ยe?mz?S?ำ?@ฉ?เด?๑พ?ฉว?6ฯ?ภี?j??Qเ?ไ?=่?m๋?/๎?JIqฟ$y?ษ>Yqพว5้=?ฆx?ฌ๏DพโฝUพกยฝณ฿T?อ?ฟ๔<|ฟHฯ;ฟข)ฟฒU`ฟFัwฟ
ฑ4พ1๗w?Zฬ๎ปdีVฟม{}?Uฟ?4?ึ:ฟ=`?์ขฟU?ฑ3พ7ฟIp?ฝ๕)=Xง}ฟ?B>ต}?'*้<ฅiuฟขำฟ๏T๓>บ๓?ล?0ใพ๘jฟฐฤmฟ3เพห<k>๏ฌC?~?o๘f?ฦ?3ผ>?พ=Q*ฟฤดeฟญร~ฟ๏๋xฟ4ถZฟXฎ+ฟๆพD๊Zพ๊`<๚๒m>Scุ>ณ?t7?U๒Q?ขe?>s?[{?๑y?ฅฏ??์|?งาw?ต์p??ฑh?_?ขนU?ฎK?้?A?I๑6?Jล,?ิ"?^1?ฅ่?3?์?>t๊์>y??>J-ฯ>ืม>Kนด>ซจ>ด[>4ม>Dำ>๒>฿ฒm>าw]>มON>0*@>G๗2>เง&>->ฃz>*>ัo๚=ท ้=^ู=o?ษ=ฺผ=ม?ฎ=m?ข=<==ฦJ=>`t=smc=?ฆS=/๘D=รM7=ย*=nฟ=0ป=z	=ฐ฿?<;๎<*?<ฆ4ฮ<Vไฟ< ซพโkkพฒu??ฎw?ทU~?/s>8{ฟn\z?9gฟ3??J?*็.พา๖-ฟบ?ฟ3จ๖พำi>ฃ๛{?p~>B?ฟก8?:>U๚ฟP,5?m/ฟ๒?๖>9Zฝฟฒ|?ย3ฟ	ฏพว?lR
พ_V{ฟฏ>sๅ?xธ>ฒ`Mฟค=aฟ?ผ๙9R?Lt?p0อ>ยฝพfฟK'yฟ"%ฟWฺฝ๓อ?>ซพQ?9ฦ}?/s?}??ยโ>๚ศ=๖!oพทฟ$ๆ=ฟรถdฟ?zฟา๕ฟ?xฟXhฟ+ปOฟำ2ฟง|ฟPโพ*พN=พnํฝ/ฦJ=5;>*_>ำญ>pี>ฆ๙>บ๊?;?ๆ'?T3?~่<?aE?ะM??ธS?ฯY?%^?ฟ๑b?ฦf?	j?? m?o?ฑตq?
s?1Bu?ฐv?L๎w?่y?ไ๐y?ภz?ฎs{?V|?<|? }?gp}?&ศ}?*~?V~?~?yภ~?E๋~?W?r0?@L?Vd?1y?B?ๆ?sจ?.ด?Wพ?$ว?รฮ?\ี???เ?Oไ?่?<๋?๎?XJฟCห>ฐปi?ำ??2ญB?ศงe?ํ?ฟมฤ>็๓ฟํr?}+พA์uฟึrฟ?fฟฅ}ฟ?>Tฟ฿ท>5ร?มพอ,ฟ)อ~?ใ๎nฟ<V?฿ฌXฟๆr?d~ฟ1s=?ฃ๑ฝ.Mฟไc?จ >@ๆฟ๖iบ=eษ? !๋=Qภnฟฎศ'ฟ=Tำ>S?ป?ณm~พ ?dฟ฿rฟ1๕พA>=?๗_}?j?U?F$>uwพfd%ฟ@๖bฟไ%~ฟยzฟฯ6]ฟุ/ฟฤํพ๗Jjพ์๒W;ฅ`>๚า>4?ญ5?ฅxP?๛d?Lr?ข2{?ฏX?ฑฤ?B-}?ส4x?Uhq?@i?n `?\V?ฬ7L?4่A?a7?`i-?Jt#?ถห?๏|?๚?๒?>ธ๊ํ>??>>ะ>dย>+ต>jฉ>f>?h>๚o>>ืฤn>ปw^>ฌ>O>$	A>Bว3>าi'>^โ>)#>2>l๛=H1๊=<?ู=?้ส=?ผ=พษฏ=Fฃ=่E=๖ท=ฝไ=฿~u=>xd=PT=O฿E=เ$8=๖]+=ยy=h=โ
=	 =ฦ4๏<P?<ฟ&ฯ<ขลภ<ั?,๖jฟ,ฺะ>w6Z?r>&?9โพภP*ฟ=bl?LยNฟN=tb|?=>6ดฃพkเพำUพ#?W}?>b0ฝ;๓uฟoแ<?ฉ๒ลฝะทพ^|?QVฟลฃ>ะBๅ=(-,ฟm??๋ฟ๔>้พฺ|?ปๅผ๓๏~ฟม+'=ฆN~?๗มธ>ำYAฟ	-iฟVlฌฝ.I?ุ๘w?ภqๅ>&cฆพfผ`ฟFi{ฟื,ฟ
Eพy๛ฬ>ฯ?L?bฌ|?u?fC?0ู์>Xๆ๕=ฃZพื ฟ๏ะ:ฟวbฟ5yฟฅ?ฟ|รyฟyViฟๆqQฟ?4ฟ<ฟ-ๆพฯฃพ	iEพ@ฝซ8.=จ>ฟz>udช>e ำ>i๗>k๒?~_?$'? h2?๊Q<?OE?.งL?ศSS?(Y?-?^?ฏฎb??f?@่i?ดิl?<_o?_q?s?)u?ปv?g?w?๑x?ฏโy?ฤณz?i{?|?7|?-}?bj}?๏ย}?ฆ~?R~?ฌ~?ฝ~?บ่~?#?.?J?็b?๔w?/?๘?ฅง?|ณ?ฝฝ?ฦ?Oฮ?๘ิ?ผฺ?ป฿?ไ?อ็?๋?ฺํ?b
ต=o?พ๛f?Dz?ร?Aฮ>%6ฟแ.W??๑mฟ;งe?ภm?>๗=ฟฏะ~ฟฟwdyฟฉeฟv	แ>q?,ไฟ๏6๏พ?q?ขA}ฟต(n?O?nฟ?5}?ํ1vฟฬ ?ืแ=๏ถ_ฟดS?๛ว>c1ฟผฆ?.M>LfฟMฬ5ฟ4Lฒ>ข'}?]*?GFพก^ฟ?uฟฆฟ}r>ฃ!6?๓?{?Q๏m?'ฝ?แ5B>๖พ\ ฟ`ฟkh}ฟ*{ฟปข_ฟ-S2ฟว๕พpyพ6ห4ผNS>Tฮฬ>cฦ?k3?๙N?อ|c?fำq?ปสz?ภ3?ึ?์j}?x?โq?ฬi?๗น`?U?V?๛?L?บB?ู>8?๒.?$?ทe?๑??(?>ฐ๊๎>D฿> ๗ะ><ร>๋Nถ>(ช>ม>๚>ข>ฎ>พึo>w_>-P>่A>54>ฟ+(>(>ชห>8ผ>น?=ีA๋=?ฺ=Dึห=&บฝ=ธฐ=[ค=๗=[]=ณ~=v=e=กU=nฦF=?๛8=*&,=4 = =Dฝ
=9=PL๐<v฿<ุะ<๏ฆม<p?~?|?]ฟร?พื+Z>๏0ฝี<jฟ็?ฝ(ฎ
?฿ผพม;โพปs??+?ฯฤ=ะ?|ฝ&g>YฒM?j๒e?wฉพYฟ6Wb?ังพ]พGสป>gนพ็ม>ถW>2Gฟbq~?^ๆ๘พ๏ฟtv?ข=o?ฟูUฝ๔ฯz?|?>ผ<4ฟ?๙oฟq>พ??ณ'{?@?>ณพ๓Zฟั8}ฟๆ3ฟ?,7พh์ผ>yH?Z{?ั{w?-G?๗>Z>Fพ๑.๙พฌ7ฟ|ศ`ฟGxฟ?ฟ}zฟg?jฟ|!Sฟs6ฟจฟขา๊พไ จพ6Mพ~กฝ๚จ=&>พt>ทฌง>ทะ>	:๕>_๙
?ฎ?๎`&?ผ1?วบ;?ปD?3L?I๎R?๏ฯX?ใ๑]?Wkb??Pf??ตi?\จl?ด8o?็rq?bs?฿u?ฺv?mศw?แx?jิy?eงz?H^{?ห?{?*|?3?|?Wd}?ฒฝ}?~?-N~?D~?บ~?,ๆ~?์?,?๑H?xa?ตv??	?ึฆ?ษฒ?"ฝ?ฦ??อ?ิ?eฺ?o฿?อใ?็?ฺ๊?ฐํ?ใฬb?9~ฟ?%>๛T?ฃ3?ทปพ?gWฟึ??tง{ฟd?้ฆ8?*Rฟพ;l^ฟ๊rฟฎOTฟO=พ๒r2?:N?ด<ฟ)มjพ`W?๛Kฟ,M|?ฝใ{ฟaโ?หEgฟW ?7{>aFnฟC@?Kฎม>ึ{ฟ?|?ฝM}?.><ี\ฟUสBฟ?e>ภ&z?&5?
พJWฟyฟ?nฟ?9ื=๐๎.?๕z?มq?๋๛%?nต_>นjพ9ฟ]ฟ\|ฟL!|ฟภ๙aฟ5ฟl?พapพาฤฯผษ์E>๗ฦ>วS?ฺ1?LuM?bb??q?ช^z?%?7ๅ?ูฅ}?๒x?ทYr?hWj?#Ra??W?=M?}6C?ฐไ8? ฐ.?sฒ$?`??ฌค??ซ?๒ ?\๊๏>Lwเ>?ั>}ฤ>ท>ำๆช>s>Cท>;ฉ>@>่p>bw`>_Q>๑ฦB>!g5>ฅํ(>๎K>'t>:Y>??=^R์=๎๚?=ซยฬ=Iพ=ฐcฑ=๔ฅ==ฉ=ภ=จ=ผw=าf=๒V=ญG=ำ9=]๎,=g๎ =hร=ฅ^=iฒ=ฺc๑<ขเ<๑
ั<;ย<zwํ>Iๅ๐ฝuwฟจซฟ(d6ฟ็ูuฟU
?BJฝแ;>๓พTฟ๑O1?tm?}?>Kฆ>ษ	?ปt?พ7??'ฟ?,ฟ.y?ๆ`
ฟpฑ=To->ร6พ"E๖ผ?>,]ฟฯw?}3ปพใ)ฟ๚l?็B>>!|~ฟAพpu?~?M&ฟคuฟlขYพIq4?J}?ฑ 
?_mพปทTฟ~ฟB่:ฟฐี[พ~ฅฌ>๎ๅB?๕ะy?/y?K??ฝฎ'>ท`1พ\๐พiy4ฟคน^ฟIwฟ๋๊ฟ',{ฟแkฟ?ษTฟ?^8ฟ๎ทฟใ๏พu/ฌพ๙Uพฅ?ฐฝ๒.๊<<~
>4ปn>๓ค>ฮ>?๓>?	?+ฅ?%%?1?#;?ง?C?พK?cR?๔vX?Gค]?ท'b?f?i?ำ{l?o?JQq?฿Ds?๖t?เnv?]ตw?ะx?ฦy?๘z?S{?v๓{?~|?1๖|?D^}?oธ}?~?:J~?ู~??ท~?ใ~?ด	?ฒ*?FG?`?uu???ฆ?ฒ?ผ?ล?fอ?.ิ?ฺ?#฿?ใ?[็?ฉ๊?ํ?s^??0ฟ>ฟ?1>๋~ไ<>จTฟฤาฏพ๖ํM?อ?;ฟe$=)v?ข=_ฟ+๋?ฟ+;ฟ?+=ฯฎb?ใ?ๆtdฟจภ<ื0?๔tฟ~ื?๛ฟฺ|z?Rฟ2ฒน>ฒ๑ล>ตxฟ?)?' ๗>a?tฟ PTพEรx?vป>@ฑQฟฐNฟง[>้v?	)@?่จฝำPฟ๙ข{ฟL์ฟว,=Jw'??w?๓ฬs?2,?e}>๒WOพI?ฟW๘Yฟั{ฟห?|ฟค;dฟEพ8ฟaแฟ 
พ"ฝ8>ม>N??/?ท๋K?VCa?ฑUp?o๎y???~?ฒ๐?	?}?DMy?wฯr?คเj?๑่a?>X?'N?z?C?ๆ9?R/?ๆP%?ฑ? 8?๕8	?hก ?ป้๐>iแ>๏ฟา>น๊ฤ>ไท>
ฅซ>?%?>{^>วE>๚า>U๚q>wa>'R>ษฅC>76>ฏ)>ฎ >?>9๖>?=ใbํ=ม๘?=ฏอ=irฟ=ง0ฒ=ศุฅ=ๆZ=#จ=ฒ=ผฺx=g=CW=ชH=2ช:=ถ-=บจ!=ฯp= =H=c{๒<ภฆแ<	?ั<iร<Sc?พใ9?GRฟ$|ฟๆฟ?ฟnp?ฟM้-?
oฟ#>^?:ฎN?k)?็lQ?4ฦ?>๊ํ>ฆLฟZ็พเํ?C9ฟฤ>?ฝo๔><ขZSพ]?ดnฟ&l?P[tพ??ฟ=`?1>โozฟXรqพL9n?ยู?ูฟ]zฟฤพj%)?ญ!??|A<พNฟ}ฟV?Aฟ?พ+>=?x?vฒz?1vO?ใA?๎=>ณกพI้็พ71ฟ/\ฟ?<vฟ`ฬฟYฯ{ฟwmฟ๙jVฟธD:ฟ๐รฟึ?๓พธZฐพ/ค]พ@*ภฝ	ฑ<ำๆ>ตh>-9ข>๛ฆห>wึ๐>	?๖ฦ?ถุ$?_b0?ื:?wC?IK?"R?X?YV]?ฮใa??ฺe?Ni?Ol?#๋n?/q?'s??t?อXv?7ขw?ํฟx?ฏทy?}z?ฑH{?๊{?๓u|?&๏|?*X}?&ณ}?๚~?BF~?i~?ฆด~?แ~?x?ร(?E?^?4t?๏?(?5ฅ?`ฑ?้ป?	ล?๐ฬ?ศำ?ตู?ื??Iใ?"็?w๊?Zํ?NY=?ถ/>*ฦ~ฟn=ฟห/)ฟ~ฟ^ฝ>6ฆ>iพาแพmpz?ศใ?>ใ๋Tพ\แพA๐zพ?ฦพ>สo|?ำ?ฏ>ิ.{ฟฎ>h ?ใป^ฟกx?มzฟ?0m?H7ฟ6?\>ฎฌ?๘~ฟGู???_kฟ@กพ๓r?:๗ใ>?0EฟhlYฟฒQ>~?p?WJ?FEึผHฟก}ฟเ!ฟซ<?ฝ?ญZu?Pv?n๘1?ฃ>ฯ3พฆฟฑนVฟ๋rzฟ๚ผ}ฟ3hfฟ?;ฟXฟพ).]ฝ๎+>?2ป>`	?-?้\J?`?้o?zy?๋ฎ~??๘?|~?ฆy?@Cs?:hk?_~b?ๅ?X?๏สN?ฑD?y.:?๔/?ํ๎%?ฉ1?Kห?ีล	?ฎ'?อ่๑>คZโ>คำ>ฬมล>dฎธ>&cฌ>Xุ?>ก>Eโ>^e>s>ฮvb>ใ๙R>D>แ7>^q*>iต>ล>4	>R >ds๎=๖?=pฮ=Nภ=?ฒ=ฆ==M=L=Y๙y=aฃh=X=ว{I=L;=ย~.=c"=6=gก=ศ?=ํ๓<ๅชโ<"๏า<ำJฤ<ดฃฟํ3|?๚/ศฝNฟๆ@ฟ
?=Rงt?'rฟtJw?=มeฟวETพ?ป]?gz?ภ๏e?ธ1x?m?๎A*>lpฟkถEพ%v?]w]ฟฐd?>X๋sพo2N>กภพต๖2?ฯ?yฟ>[?ว๑?ฝSฟNP?;<ศ>ใsฟฆพ~9e?๑@#?$ฟC}ฟฒฌพx;?้?แล?U	พ๔Fฟ?๑ฟHฟ$พp>8?v?พ|?K7S?Vc?ณT>ฦัพ%฿พs็-ฟBmZฟ~!uฟb?ฟg|ฟFnฟฤXฟ%<ฟ?หฟfn๗พดพตฐeพsฯฝลมo<๛๚=?ญb>m}>๗0ษ>{ข๎>เ	?่??$?ด/?๒9??๐B?ิJ?aปQ?๊รW?]?a?e?์i?."l?ฤn??q?
s?รt?ขBv?๛w??ฏx?9ฉy?๓z?ำ={?ชเ{?ษm|?่|?
R}?ืญ}?`?}?FB~?๖}~?ฉฑ~?p?~?:?า&?๊C?]?๑r?ื?6?cค?ชฐ?Lป?ฤ?zฬ?bำ?]ู???ใ?่ๆ?E๊?/ํ?eฦOฟผd?g!8ฟฺฝ}ฟนฟ~ฟe1ฟ0๘O?W~พjฌ>)Tฟu~D?ฐT?๓ณW>2jฝ๊G๐=R)?(}?Lทf=ศ0ฟ>?F>ฝต=ฟ๙f?flฟยiX?-ฟฃF={ญ!?F๊ฟv7๋>8+?_ฟkหพaHi?a?ึh7ฟใ๏bฟUฃ=ใj?ื&S?|V๖<?c?ฟEฟว๒)ฟฺญผ ล??r?gx?ณ7?Ox>#พส8ฟ!\Sฟฬ7yฟลa~ฟ8hฟํ>ฟ๓	ฟG#พโฝ>NEต>!฿?โ+?๎ศH?ๆ๕^?ลn?y?N{~???1F~??y?ตs?*๎k?nc?$zY?`mO?#&E?iา:?0?&?Iส?/^?yR
?ลญ?็๒>๒Kใ>ิ>ถฦ>?xน>'!ญ>ก>ถฌ>ต~>ท๗>ฃt>nvc>่S>]cE>ตึ7>23+>j>m>-0
>ฅ >เ๏=_๔?=ฮฯ=ฃ*ม=สณ=mVง=4พ=็๒=ๆ=๔{=(ฎi=แxY=ไbJ=fX<=๔F/=]#=ห=ศB=๗t=vช๔<
ฏใ<:แำ<,ล<ฟx?ๆ>ฺ1?ทพน9สฝ8?ิG?่tvฟ`t?ฟ$ฟึV?Az??b;~?G/@?3พ๔ฟปข=Nw\?)ouฟาใ+?ห?พ๊yฤ>ภภฟ.ฌN?N}ฟ0|F?m๋า<มbcฟ>?ๅ๙>eๆjฟe?าพ้Z?เ2?๑์พ	@ฟหพพ?]โ??*?๙ฒฝs?ฟ>๑ฟ )Nฟ๎ฃพ-du>{L2?้s?ฅ(}?IีV?ธp?g$j>`ๆๅฝ๋Lึพ8*ฟ0Xฟ๗sฟ๔fฟT๓|ฟ%koฟ/Yฟ??=ฟะฟy๛พ งธพjนmพ๒ธ?ฝ?๚;gํ=?ฃ\>cภ>นฦ>m์>๐?{?ๅM#?E/?ณX9?jjB?:^J?ETQ??iW?น\?%[a? de?็h?๕k?ๆn?๋p?์r?ใฉt?],v?ช{w?x?ฒy?\uz?้2{?4ื{?e|?๙เ|?โK}?จ}?ม๘}?F>~?~z~?จฎ~?ื?~?๙??$?:B?ง[?ฌq?พ?C?ฃ?๓ฏ?ฎบ?๘ร?ฬ?๛า?ู?=??ฤโ?ฏๆ?๊?ํ?ณnฟe?นg=ณ1HฟฺคKฟ๊ฅฝs่?Eฟ๑B?kฟปม>mt}?3ข?ณภฅ>ฐํ>_?\ลd?wUqพO.pฟษ}5?อ|Q=.|ฟฝK?IpUฟPฯ<?ื์ๅพfยนฝส]<?Gฤ|ฟIฑ>M@?ึPฟ๐n๙พz^?ว?p(ฟา,kฟL?;ษc?[?ธฐ=X<6ฟชฮฟZr2ฟ พฝ?^o?A{z?ย>=?มฉ>?ต๘ฝเณฟ%เOฟ?wฟ๋~ฟjฟ ๎Aฟ๒ชฟขขพด%ฉฝํ>Pฏ>Y?ซ)?ฮ/G?มว]?๖m?ุx?D~????'v~?รPz?่$t?rrl?ฅc?>Z??P?อษE?ถu;?71?ณ)'?b?ส๐?ใ?
?ฌ3?	ๆ๓>=ไ>ฺkี>woว>ปBบ>฿ญ>ะ<ข>นS>>>./u>?ud>6ืT>BF>ฆ8>?๔+>ะ >๔>"อ
>ะ7>X๐=(๒฿=)tะ=ฝย=ด==จ=ูo=G=w=6|=ํธj=/qZ= JK=/==&0=ฎื#=y=(ไ='=?ม๕</ณไ<Rำิ<kฦ<Iผ>ไพ}?}?"!?ฅi?6?ผw#ฟ_l&?ช๚ฝQ๗lฟ	>BN?น6r?พb?ฆ๙>ศๅพ๐ษxฟำ0ฑ>อ4?;ชฟ#?Q?-ฟนY?ฯเ,ฟผd??๑~ฟy\-?ๆ;">บ1pฟ฿๛(?+?ๅ_ฟกL?พ'N?ศ@?Fสพ?ฟQช้พjบ??tห3?ภ็ฝR7ฟ\|ฟฅ๔Sฟตพ๑xS>ศf,?q?๓~?OZ?Ni?v
>bผฝF`อพ.'ฟขใUฟ๎ฝrฟ ฟt}ฟYpฟ-"[ฟี?ฟ?ฯ ฟ๙บ?พ?วผพ+พuพ๛ํฝ๛1:ญ0เ=V>>ธ@ฤ>.6๊>K?6(?"?RW.?ฯพ8?VใA?ใ็I?ย์P?qW?งj\?ea?!(e?๔ฒh?ฦวk?un?`ษp??ฮr?t? v?Dhw?ซx?y?ถhz?๒'{?ดอ{?[]|?ึู|?ณE}?'ฃ}?๔}?A:~?w~?คซ~?:ู~?ถ ?้"?@?/Z?fp?ค?N?ฝข?<ฏ?บ?nร?ห?า?ชุ?๐??โ?uๆ?แ้?ุ์?#Fพจป4>PM?;ผฝf'-พgึ??ซG?ณ ฟัL}?3?eฟใ@สฝqฃs?$ฦ^?4)?฿๚??ฐT|?รฒ5?MๆฟมEOฟy$]?V=พ๏6ฤพ$ฐ'?ฟด6ฟd@?&mพ?2xพ1S?J!uฟh>,~R?ยภ?ฟธiฟำพQ?c
)?r_ฟ\rฟu๔ฝ>น[?c?xx>๗,ฟห?ฟ!:ฟฐสืฝ#?Sๅk?"|?๘B?ๅท>ี๔ภฝผ ฟ;FLฟdvฟ๋Xฟ?klฟ฿Dฟ0ฟใชพr`ฦฝว>?Tฉ>ฯ?qr'?E?ฑ\?#m?
x?	~?ณ??]ฃ~?ขz?ฦt?๕l?h6d?3ฑZ?kฏP?ฐlF?`<?ื1?rฦ'?~๚??k?cน?3ไ๔>ั-ๅ>mOึ>Fศ>ดป>ุฎ>๋๎ข>ซ๚>jท>F>ง@v>ue>ฮลU>ษ G>Fv9>ฦถ,>|ำ >\พ>j>ส>หค๑=๎๏เ=`ั=ิโย=pdต=ิจ=~!=ง==h='U}=ฒรk=|i[=1L=>=Wื0=?$=j&==Vก=ู๖<Sทๅ<jลี<ท๎ฦ<*{?)๛{ฟ6?๊~?BP|?าืN?%6 ฟดฝ๚S>@hแ>ฃฟ~ฟิ-พ{@?>ผ@?ฮY)??ฟ,>|V4ฟI\ฟ3??๗ ?๛{ฟztl?oAฟQ3?Kฟ.t?ฦ]xฟ๔ฎ?u>Oyฟpณ?)??Qฟ_ฦฟะ@@?๓ถM?wฆพ๔}ฟ3^ฟIv์>Tg}?ฅ=?v"<จH/ฟ0~ฟฝlYฟ~ใฦพ2O1>๓T&?๎๊n?z?~?ฅ]?cL?ย๒>&ฝm`ฤพฃ#ฟ?Sฟvqฟิห~ฟ?้}ฟqฟฏฅ\ฟคAฟึห"ฟf์ฟๅภพืพ}พะ:?ฝvฮปR๗า=P>B>ฦม>ไ?็>๑?DG?ภ!?วง-?^$8?ร[A?qI?ุP?ฌดV?s\?\ั`?์d?ค~h?Hk??Mn?งp?	ฑr?:vt??u?ศTw?ล|x?t}y?\z?๏{?(ฤ{?U|?ฌา|?}?}?ฦ}?t๏}?76~?s~?จ~?ึ~?p?~?๑ ?ำ>?ตX?o??Y?่ก?ฎ?oน?ไย?ห?,า?Qุ?ข??=โ?;ๆ?ฏ้?ญ์?L8?}/ฟqy?C'?SJ?a|x?ฝx>1ซ^ฟฌj?๒ลฟLฟค8?๎แ~?oฯe?^ฬo??K|?ฃ6้>@L<ฟ๏๊ฟดv?ฐ9าพซโ/พBq๙>`Rฟ็้>ง	พาฦพ๔ฐe?$iฟฝึ=?เa?ซt,ฟwฅ&ฟ๛2C?่9?งQฟฅwฟ'พUผR?ุษi?ุ5J>ขt"ฟฟุSBฟบพi?>h?Z~}?]รG?้฿ล>ยฝะ๔พ๊Hฟธฬtฟ*ซฟ Anฟ๓ฟGฟซฟnฑพยใฝ_๛้=ฒRฃ>?>E[%?M๎C?ฝ\[?๓Jl?w?ส}?9๚?ิอ~?๒z?ฉ?t?	vm?Qฦd?K[?OQ?สG?eบ<?mw2?รb(??%?๗?๊>?โ๕>bๆ>อ2ื>}ษ>ึป>Zฏ>๐?ฃ>ก>ฏS>|ฎ>Rw>๓tf>YดV>q?G>F:>x->#!>มf>>D\>:ต๒=ฑํแ=ุLา=๊พร=^1ถ=ูฉ=!ำ=ใ=Yด=พs~=uฮl=ศa\=6M=ฐ?>=1=PL%=ะำ=่&=7=๑๗<wปๆ<ทึ<ะว<2`1?6ร:ฟาcพ3A?#V?๛Fv>ZWxฟ?>\ฯพS}T?NUฟ๖R1ฟฒ%ฟ=:แ>a@ณ>?-พฉๆcฟo-ฟกณH?๛>ฺliฟ2|??_ฟ:มR?มอcฟหญ}?่่kฟBโ>๙{ำ>พ~ฟQ๔๐>2=?	UBฟGก%ฟซ้0?	PY?นพ	ฟ}ฟxYฟยะ>๕z?PูE?dืo=?ฆ&ฟ$6}ฟ"^ฟ๛ืพ
๑> ?เl?m?ฒึ`?B?ปษ>
XPฝ9Nปพฮ ฟQฟmpฟ*j~ฟแR~ฟD?rฟฉ!^ฟMnCฟร$ฟo๘ฟฅ?ฤพฅ?พ2;พ๓Yผฐปล=6zJ>ฝ>๘Jฟ>1ฤๅ>ไ?ฅe?ู๘ ?ค๗,?b7?ฒำ@??๙H?P?YV?๏ห[?`?ฆฏd?Jh?lk?J&n?p?r?;\t?๛่u?6Aw?ฬkx?ผny??Oz?เ{?บ{?วL|?yห|?A9}?^}?ฦ๊}?)2~? p~?ฅ~?๗ำ~?(?~?๗?=?9W?ึm?l?b?ก?หญ?ฯธ?Zย?ส?ฤั?๗ื?T??๚แ? ๆ?|้?์?ำ?x?^~ฟ ฅ์>Hช?@x?
๖e?JWษพศ๎?พพ?็ะฝฟ้`ฟ็ฝฐ>dUr?fz?)??h_?)ฮ>์fฟญpลพ๕?๎ฟ7มD=ฺ>4Kอพ๗P>B๛<skฟ๎zs??Yฟ,ผAn??+ฟ๔69ฟธ๖2?oคG?
ฦ๊พdฯ{ฟ Nพ?H?,ฉo?ศ'>pโฟ3~ฟjญIฟKAพZ๋>ฏd?ณ~??นL??ฎำ>ภ"ฝ
G้พบบDฟgsฟัแฟ pฟธJฟวฟ2แธพZ พิฮ=๘I>n\๙>4?#?FB?์Z?Tnk?๖v?@}?๒?๕~?4?{?hu?V๕m?ืTe?ฉใ[?ฉํQ?ฐG?ล[=?ิ3?ฆ?(?L)?ๅฆ?ผ?@ฤ?฿๖>ด็>๙ุ>ย๒ษ>A?ผ>ฐ>เRค>YH>ๅ๏>ฆ@>acx>Wtg>ูขW>?H>ธ;>@:.>ล<">">ํฃ>y๎>ฅล๓=p๋โ=+9ำ=?ฤ=J?ถ=ฅQช=ย=c=JN=T=8ูm=Z]=P?M=ศด?=ธg2=?&=5=Hศ=ณอ=๙<ฟ็<ฉื<Nฑศ<;๗mพลๆ=ำcฟลlQ=Rt>ง๚เพ	`kฟruf?กTฟg?๔พyCpฟฅพqด=ฺ๕ป:๚พ?|ฟ,ป?พ&3l?ไ<3JฟYด?cLtฟj?*฿tฟแ?น?Yฟ$(>อ?๔ฟLป>ุN?uน0ฟF7ฟ? ?c?ซ8พ๋รzฟใทฟF?ณ>ธw?QN?L?=xญฟืe{ฟอYcฟ๗ศ่พBัุ='ณ?{i?งฬ?wโc?9ฯ#?	?>ั๘ผ*ฒพฟ)ฅNฟEบnฟ!๛}ฟ๏ฐ~ฟๆsฟ_ฟM2Eฟ?ถ&ฟฟgษพฒูพืพ๕๏ฅผ๋}ธ=mhD>มฟ>ฮผ>ใ>&?Z?0 ?๊F,??ํ6?"K@?0H?าณO??U?|[?uF`?
sd?]h?บ>k?k?m?ๆap?ur?Bt?Sาu?-w?ภZx?๔_y?nBz?ฤ{?๒ฐ{?oD|?>ฤ|??2}?๑}?ๆ}?.~?yl~?ข~?Rั~??๙~?๚?d;?ผU?l?N?j?<??ญ?.ธ?ฮม?$ส?\ั?ื???ถแ?ลๅ?I้?U์?ิ๐จ>า ฟ'ภพไ}:?๘_?ูฯ>~^ฟ?>ค๎<n?เ>?ฟงๆอฝcS;?uNr?mn?๗(??ฉ"พPๆ|ฟ1S๙ฝB฿x?QHฟG๐> นศ=๐`พดQ๑=>G><?&ฟG|?}Eฟพฅuw?]# ฟฏ๎Iฟ.!?๑ภT?ฒbลพๆ~ฟฏบพ์">?จt?ฬ>สๅฟ[ไ|ฟ๛Pฟฉkพ๕Tู>์_?ฮU?{{Q?ไNแ>v๓Gผณ?พ8ส@ฟศAqฟุ?ฟซจqฟQMฟ?ฟฝ5ภพ๛ๅพJคณ=
;>@.๔>Q!?ฝ@?F?X?0j?ํhv?ZB}?ฒ็???{?zะu?๗rn?๘แe?({\?YR?คPH??=?ดต3?)?,ภ?\8?8?fI?ู?๗>ล?็>๒๘ุ>?ศส>ีiฝ>ีฐ>นฅ>๏>>ลา>กty>ซsh>LX>ฃผI>eๅ;>๔๛.>b๑">~ท>ิ@>ฌ>ึ๔=+้ใ={%ิ=wล=4หท=pซ=c6=ฟ-=9่=uX=๙ใn=^R^=iๆN=฿@=่/3=๐ภ&=.=งi=โc= ๚<พร่<ฐุ<ษ<ฉqฟ:]?6mฟQ`/ฟ}๙พ๔๔iฟ?.?พt๚|?0ไฟ๕e?ำ3ผ๒ณ~ฟ4|.ฟน5ฅพฟkบพศQ@ฟ?ฟ|ฟXโพ~?พ๔oพจฟแฯv?zฤ~ฟDฟy?s7~ฟฉ{?ฌBฟ?->?r#?มa}ฟ1%>i]?ุYฟยใFฟa?6l?ูฝOvฟk+ฟต>.ณs?ใผU?&>3aฟ#yฟฺสgฟaG๙พY=k&?jแe?๛?Yศf?m(?S>ซ>pี ผ%๖จพ่ฟอLฟFmฟฝ~}ฟeฟ๎tฟัaฟx๐Fฟ/ฆ(ฟฎฟL&อพำพ?pพ^฿ผ)>ซ=อT>>?>?Oบ>?Lแ>ถ?d??ฃg?+?สQ6?ย??
H?ถJO?6ขU?๓+[? `?/6d?fเg?ชk?aึm??p?ฮVr?็'t?ปu?ัw?ขIx?Qy?5z?๛z?Fง{?<|?๛ผ|?ฒ,}?~}?Yแ}??)~?๎h~?q~?ฉฮ~?๗~???ช9?=T?@k?.?q?d?Wฌ?ท?Bม?ซษ?๓ะ?Aื?ท??rแ?ๅ?้?(์?bถฟ7์ศ>ดsฟ๗&(<5!>sfพฯฤ}ฟ|0?ฟฟq\T?๐cฟฆฟg?ฦ><L@?pk=? fพ>E๊พ}ฟlุ>ถa?Whฟฐ๐>ฐ?ฝำ๘ผฤytฝ}+ต>ฦ#BฟW็?]-ฟ3พ็]}?Bฯพ็กXฟ< ?FE`?eทพเฟ.วซพฆ2?9รx?๑ท>dฟจzฟๆWฟจพ.๚ฦ>ี่Z?oะ?ทV?๙ผ๎>xฒx<มัพ๖ฝ<ฟOoฟ<?ฟก:sฟู Pฟ?ฟ~วพๆiพ?k=&&>ฝ๗๎>ช๘?ๆ>?ะW?งi?ฏืu?ฯ๘|?ฅู?ถ<?lา{?g6v?ํ๎n?ตmf?]?(S?a๐H?>?T4?5*?ฑV ?ษ?x?[ฮ?ศู๘>๎่>ถ?ู>อห>G3พ>๔ฑ>|ถฅ>ภ>'(>ุd>ฯz>๑ri>ณY>,J>
ต<>กฝ/>๙ฅ#>ึ_>น?>?>nๆ๕=ใๆไ=ศี=Sฦ=ธ=8ฯซ=่=ำ='=ฟ็=น๎o=จJ_=อO=๖bA=๘3=@{'= ?==๚=ฆ7๛<โว้<วู<ไsส<qฆIฟ็vk?ฃผ?พ?ฟzฅsฟ/vฟใๆ>?ไ9?kv\ฟี๖?ั๊>3บYฟฆ฿kฟe?(ฟฏ5,ฟคmฟ
cฟZ>b}?;๑พยZุพ๓a?ๅ~ฟรแ?)ฟpo?ู?&ฟตเ<ฌ1<?ส้vฟ)~>oj?ฟgฟฆUฟ่ใ๖>cs?e}?ผt)qฟ*e7ฟฤq>่n?งะ\?x!P>๑ฦ
ฟ๚nvฟเkฟZธฟญ=t?ewb?s๘?฿i?ภ๓,?Dูต> /<๛ฑพฤ;ฟ!Iฟฤkฟ๕|ฟ?JฟQuฟๅgbฟมจHฟ*ฟฮ

ฟA4ัพหพพ ฝ?=f?8>F8>Tะท>ษ฿>?ลผ??ณใ*?/ต5?8??~G?4แN?FU?|?Z?oบ_?๙c?8ซg?jโj?-ฎm?0p?y8r?t?นคu??w?p8x?3By?ฃ(z?h๐z?{?ฃ3|?ฐต|?a&}?}??}?ใ%~?_e~?\~??ห~??๕~?๛?๎7?ผR?๓i?~?w??ซ?๊ถ?ถภ?1ษ?ะ?ๆึ?h??-แ?Oๅ?ใ่??๋?)ๅ~ฟพุx?UฒZฟRฺ6ฟw#ูพ*ETฟ๙ธ5ฟ๋uz?9กcฟc?-3ฟbXฟ๐wธผโ>๏?ๆ>=น(=56ฟxfhฟ.hำ>ว๙;?ี๐zฟHโ&?\mพฟ?#>A๘pพส ?pYฟH~?X?ฟIรพ-ๆ?Oูพ +eฟษ/๓>Vj?ฤnพ(มฟ?ฬพU&?q๕{?า>l๋พำwฟพ/]ฟPพ๙Pด>๋U?r??)]Z?#๖๛>*J.=zษลพ8ฟ>mฟ?฿ฟ?ตtฟรRฟK-!ฟaปฮพๅ+พ?Xz=>น้>Pฮ?s/=?LV?jฝh?[Bu?ซ|?hศ?*\?|?Vv?6io?๘f?ฌฆ]?ูรS?UI?<??ฺ๑4?ดฯ*??์ ?kZ?|$?S?hึ๙>'?้>Gพฺ>tฬ>?พ>6Pฒ>(hฆ>X<>2ฤ>เ๖>้{>'rj>nZ>ญyK>ง=>G0>Z$>*>z>
ฅ>ฬ๖๖=ไๅ=?ี='/ว=eน= ฌ=??=vx==w=y๙p=๑B`=ดP=:B=Fภ4=5(=e=fฌ=>=-O?<ฬ๊<?ฺ</Uห<0พ=ร`p>?ว*3ฟsึgฟdฟ:P4??S>JE๊พง=%R?อฟb๏ฟฏeฟdฟ`ศฟฮา3ฟ3ธึ>(i?ๅศ-ฟดJพา B?/sฟnณ|?ฯxฟฏ>]?&ฟฤ์ฝQ?ฐฅlฟ๓ๅ<ฺs?{1ไพฌGaฟํ+ฯ>ว?x?mU3=jฟeBฟ;5>\i?งIc?;X>พใ ฟญJsฟ@oฟทฟF;ช?+?^?ฅฤ? l??`1?]ภ>]??<ใ^พ?ฟUไFฟ44jฟ^|ฟzฟavฟ?ลcฟ[Jฟw,ฟ?
ฟ2>ีพ!ภพํ$พญ(ฝ=น=F(2>ัr>Oต>ฯ?>ฬ ?}ุ?๊ำ?71*?
5?ฎ>?|G?LwN?w้T?ดZ?t_?ปปc?ำug?๘ณj?ฮm?๙o?r?!๓s?ฦu?๒v?,'x?:3y?งz?'ๅz?ฯ{?/+|?]ฎ|? }?}?ุื}?ร!~?หa~?C~?Oษ~?์๒~?๘?06?9Q?คh?์|?|?ณ?฿ช?Gถ?)ภ?ทศ? ะ?ึ???้เ?ๅ?ฏ่?ฯ๋?๖t๋พ,ูD?-T!พEแฟWumฟM~ฟปฬพ3l?ฟ๛f?W๐๖ฝ{o~ฟE?พฆ?=ฦรั=3กพFeฟgE?ฟั$?๕
?ฑะฟATM?w?พQoฐ>9ฮพฃy#?๚kฟQsw?]?๋พฎ]?พิ?HMพ๋ioฟaDศ>B0r?>พ/2~ฟ4ํพ[?0<~?Jh์>๑~ำพชhtฟUฬbฟ*ฏณพฑ`ก>/คP?ศโ?ถz^?น{?!#=ฐนพT4ฟukฟ"จฟ4vฟ?-Uฟ-r$ฟ๋ีพ$X:พ?ฮC=p๋>lrไ>S?s;??T?ัฮg?๒จt?ฬZ|?๛ณ??x?:\|?F?v?ัแo?๚g?ฏ:^?ง^T?|-J?ษฺ?? 5?ฺi+?ช!?๋?Cฏ?ฒื?ธา๚>wอ๊>ฃ??>/Jอ>ฤลฟ>]ณ>พง>?โ>/`>?>๑ง|>Mqk>\\[>#XL><T>>็@1>%>zฐ>w>57>%๘=Hโๆ=Z๊ึ=1ศ=ๆ1บ=ฦLญ==Kก=ฯ=ถ=P=7r=9;a=ฑQ="C=u5=?๏(=ษ6=ฤM=l&=ดf?<(ะ๋<๕q?<y6ฬ<ืRc?ขช#ฟอ|?ึ๚<kNฟพ-๛ี=ก}?*qลพย#ฒ=เพ?!~?%โฝ'gฟซuฟUง~ฟฉซtฟฆ|ไพจ&*?โC?ฟWฟร=Xไ?๔m^ฟ๊Qp?;jฟ?E?Uศพร6พGc?฿ฝ^ฟอฦฒฝอz?ศKตพฃkฟ{ๆฅ><๘|?ภณ๒=`?bฟ๛LฟV๐=ซc?[#i?ิQ>yํพขทoฟ๓rฟWฟ2๔ผ7J?>[?ฝ_?n?ดด5?ีษส>T=ฝ?พ.พฟ2Dฟhฟผน{ฟตฟ9wฟาeฟ|LฟฎY.ฟำฟDูพัฒพ๕0,พ	;Eฝ\t=|,>?ฌ>oอฒ>ฺ>จ*?>๓?	?'~)?]z4?๚#>?F??N?T?9Z?K-_?"~c?6@g?Vj?E]m?ๅีo?m๛q?ุs?ปvu??v?ีx?1$y?z?ฺูz?{?ฒ"|?ง|?จ}??|}?ำ}?~?4^~?'~?ฦ~?๐~?๒?p4?ตO?Tg?ษ{??ุ?"ช?คต?ฟ?<ศ?ตฯ?.ึ?ษ??คเ?ุไ?{่?ข๋?มZ?>oZฝo&??+ฟอ๔nฟ{ๅ1ฟถ๙>G
?โKฟฑ'?rHท> qฟ๎?Bฟฒชคพ|ๅพกฟWC}ฟฒฟฆS?>_vฟใะi?m)ฟฐA?ฟศฑA?*xฟok?๗ญพZศฟุฟz?C มฝrDwฟT>tx?lAฝq5{ฟ"ฟป?V?ว?:ับพnipฟท๑gฟpผวพอ0>!K?{z?P_b?฿
?ฒว=รxญพ๘/ฟลhฟฌTฟgwฟ;ชWฟซ'ฟำ?พ๓ภHพำ;=6}>๙#฿>ฤk?หฒ9?เงS?ล?f?yt?X|?]?ฬ?|?5\w?พXp?h?อ^?}๘T?ุสJ?่x@??+6?,?"?Q{?อ9?\?ทฮ๛>ผ๋>ห?>?ฮ>ะภ>hสณ>>หง>S>?>ห>ๆธ}>dpl>J\>6M>ษ#?>2>ร%>ลX>Qด>]ษ>z๙=๕฿็=ึื=8็ศ=ษ?บ=ฎ=ู?ก=(ร=๎O==๔s=3b=วR=7่C=ฃP6=,ช)=.ไ=#๏=ผ=;~?<Jิ์<d?<ฤอ<vใ]?ขฟญB?>?รฑท>?8?}_?9qWฟ๑ฯ?;Tฟ	o?%๙ซ>ธ฿%ฟfrฟำ9wฟ:MฟVพUPZ??|hsฟX>*{ะ>ํ้?ฟ0[?]CTฟc'?Sb{พแศพํp?iMฟ_TNพW?~?Zพฎใsฟลv>bO?ศ/E>๘ZฟคVฟbi=\?จYn?๚๎ฏ>฿ฎุพ{ทkฟ๎uฟฬ?ฟสฉฝD๏>]W?ะษ~?ั?p?B๎9?ุี>ป?=iพ๎	ฟsAฟ๋่fฟ9{ฟูฟGxฟhfฟำญMฟP70ฟขฟนE?พ ฃพม3พฦaฝ"\l=๕%>ไ>Jฐ>0Mุ>a&?>๗?ด=?ส(?&?3?๗=?$%F?LขM?M/T?3่Y?Nๆ^?K@c?a
g?Vj?4m?ฒo?ถ?q?๊ฝs?_u?สv?kx?y?z?ฮz?-{?+|?|?B}?sw}?Aฮ}?t~?Z~?~?้ร~??๎~?๋?ฎ2?0N?f?ฅz????dฉ??ด?ฟ?มว?Kฯ?าี?y??^เ?ไ?G่?u๋?sฒ?ล+Uฟธ??PU=ฬเพ]/ฝj?๓	ฝ7ๅณพL}=Dุ@?๔$3ฟฑอuฟ6ฦ(ฟขZฟา[TฟบP|ฟBพูตr?ฦv=O#_ฟ๖z?ๆ?Lฟ{+?f2ฟ&Z?v?~ฟgฯZ?q?VพU1ฟอ"s?!TV<ฆ|ฟ;[>$?|?ตT๊;;ฯvฟฟ๛>ก??(๑??กพGุkฟ.lฟอn?พธu>มBE?ชฦ~?๗	f?#?ย?="$กพF+ฟก\fฟฆๅ~ฟรxฟaZฟ#ู*ฟข$ไพ5WพดDญ<7q>้อู>ด3?Oํ7?yNR?Kไe?๑is?Bฎ{??๙ฉ??|?%บw?๛อp??h?3__?\U?ggK?_A?ศ6?ำ,?4ญ"?S?ฤ?Dเ?gส?>Uซ์>พd?>็๔ฮ>ธWม>Vด>ง|จ>ด/>?>ฏฌ>วษ~>lom>า8]>๐N>M๓?>ฤ2>"x&>>(Q>[>ส'๚=?่=เยุ==รษ=ฉหป=Mสฎ=sฎข=h=ฺ้=เ$=ฐt=ว+c=?iS=KฟD=ะ7={d*===ศR	=ม?<lุํ<"V?<๙อ<์3G=๗ภฟืน?ป~X?ิf?b]???ฬ>าฺฟญo?๓^ฟ](?เ6?ผพญ|@ฟ?1Oฟย๗ฟ->ตw?:าข>ฟ?๖๚>V"J>aUฟo>?ฯ7ฟ1?ฒLภฝCโฟSKz?๋8ฟแB?พd๚?%พqzฟไ>S??๚๔>ุ)Pฟ,_ฟvร\ปC`T?็่r?X!ว>Drรพ	LgฟxฟQ+#ฟ(ศฝฉเ>แR?๛~??r?>?KU฿>:็ฝ=)tพฒฟฆ>ฟ2.eฟIzฟW๑ฟภหxฟwฎgฟNOฟp2ฟ@๘ฟ&CแพพzO;พ<O~ฝ?ฬQ=#ู>?>ลญ>
ึ>ว ๚>ป'?ฌq?J(?h=3?x=?ฯชE?47M?ฐัS?zY?^?5c?Vิf?'j?ฒm?o?฿ฝq?#ฃs?[Hu??ตv?๏๒w?ํy?a๔y?รz?Mv{?|?3|?ิ}?แq}?nษ}?F~?๚V~?ไ~?1ม~?ๅ๋~?แ?๊0?จL?ฐd?y??!?ฅจ?Zด?}พ?Eว?฿ฮ?uี?(??เ?_ไ?่?H๋?Sก?2qฟ๐J%?เD?S>ยF?Hัx?จwฟฅฤN>ฆ*เพว6y??กพZ}ฟeฟnVฟ\?wฟ|Xbฟร่=สฎ?๗ FพHj;ฟ.๐?๕gฟDvL?Pฟfm?<ฟE?_๏ฝPFฟจHh?+F๖=Nฟรซ๚=ก^?ญsณ=โqฟm)#ฟ๗?>ฌz?ง?ฅ?พ๛ทfฟHฬpฟHฝ๎พ
aN>,??ว}?ปyi?ุG?ร&>ธดพK๕&ฟ~ืcฟ[~ฟฝผyฟเn\ฟร๚-ฟ,๋พ'reพภ <_ูd>hpิ>2๗?#6?i๐P?f่d?]ฤr?R{?c?dพ?;}?x?Aq?Ui?ฒ๏_?A)V?)L?-ณA?ธc7?ฆ5-?๏A#??+N?Cd?ฦล?>โํ>{F?>สฯ>~ ย>)Dต>๚-ฉ>ึ>ฬ3>>>ฺ>dnn>๛&^>H๓N>สย@>?3>,'>Pฉ>๛ํ>คํ>8๛=E?้=ฏู=@ส=ผ=ฏ=`ฃ=ื=ฤ=&ด=l$u=$d=๓PT=_E=?เ7=ศ+=๕>=เ1=๕่	=ฃV =?๎<8H?<Yฺฮ<PoPฟฏฌ>|Cฟฬก#?ุ?t?F@O?0?pพฅMฟบz?ฬ(fฟษ5j>ูr?a>ฟโพๅึฟจพ2๏>
ฮ?๎ุK=){ฟc.?๚ณผฌุพท?ุ%ฟ๊ธพ>ฒpu=4ล"ฟ?7??!ฟ9ืพp$~?h}ฝ>~ฟ่ก=๎?nฌ>RDEฟํสfฟ๖ฝL?ๆอv?ฺุ?>๏ฮญพPwbฟ1ฟzฟ%C*ฟณพs า>่N?b}?๎๖t?yB?ๆq้>9ผ็=}aพ,ฟกห;ฟecฟฅ}yฟ??ฟjyฟr์hฟ8่Pฟๅ3ฟก๋ฟA<ๅพ{ขพฟฺBพajฝ;7=ฑป>.ค|>ป?ซ>ลำ>?๘>?@?ฅ?a'?"2?~<?
0E?ธหL?นsS?qDY?W^?เรb?f?L๘i?จโl?\ko?็q?As?1u?กv?`แw?ณ๖x?-็y?ฉทz?al{?	|?ภ|?_}?Il}?ฤ}?~?WS~?พ~?wพ~?้~?ี?$/?K?[c?Xx??D?ๆง?ดณ?๎ฝ?ษฦ?tฮ?ี?ฺุ?ำ฿?#ไ?฿็?๋?Y,พพ๎พ>'พrb~?v]?nQx?ะย!?>Orฟ\i1?Tฟีคw?ู?>?Xฟ?pฟ;zฟฟ์ํ1ฟศง>้y?ื}?พ?3ฟ-x?ษ+yฟ??e?ซ*gฟ๔	z?ฤ)zฟผQ,?Kๆa=YฟlVZ?=ug>ะฟุ๒<.๗?๒+>ธแiฟ๙0ฟX ฟ>๕~?ส฿%?ห๚[พฅaฟั|tฟฯ ฟ฿&>ิ8?d}|?ผญl?คJ?ตฮ6>,พFO"ฟ๓5aฟต}ฟ^ฤzฟถ^ฟ11ฟG&๒พ	นsพ|ฯดป-rX>จฯ>Qถ?DT4?ทO?่c?ภr?:๓z?hB?ะ?S}??ox?dณq??i?`?-ภV?L?ROB?ี?7?ฮ-?Mึ#?w*??ื?่?ิภ?>-๎>(฿>๔ะ>!้ย>เ ถ>5฿ฉ>A|>ฯ>Sะ>จu>Mmo>_>ัO>>A>&G4>แ'>Q>ส>ฤ>]H?=็ุ๊=[ฺ=@{ห=eeฝ=ฮGฐ=ฅค=,ณ=ฎ=mC=&/v=Re=8U=smF=+ฉ8=ู+=Y์==ำ="
=fโ =ฑเ๏<N:฿<ฃปฯ<ฏmฟฃิt?|ฟ^ถๅฝฟL ?R๘x>อjFฟ1ฅพ~8?Xฟฟพฬ}?E?ะHฝ+Xพอ)ฒ=ณ8?์่q?9cพบfฟXU?ฉ	vพ๐jพ}าเ>F๛?พWฑ[>๎jY>์Q=ฟA?9ฤฟฎจฟ็_y?ึ=3ใฟแcผำa|?/ะ>๚h9ฟ~mฟษร?ฝ฿?B?๋z?ฑ๔>>ะพ;]ฟ๕|ฟฺ 1ฟ	?(พ"ร>๚I?7ใ{?วv?๗๗E?hq๓>ผ>^?Mพw?พไ8ฟ!aฟชคxฟ?ฟ?7zฟy"jฟ-|Rฟ๗ด5ฟป?ฟ๖0้พ?cฆพ6cJพaซฝ3จ=ฯ>w>ฝธจ>ฦั>ค๖>WY?วื?!ฌ&?T?1?๕;?ีดD?ื_L?iS?๒X?ฎ^?Mb?gf?่ศi?uนl?Go?ฮq?Ams?u?Jv?พฯw?i็x?ู๋y?*ฌz?kb{?` |?E|?ไ?|?ซf}?ทฟ}??~?ฐO~?~?นป~?(็~?ว?\-?I?b?0w??f?%ง?ณ?^ฝ?Lฦ?ฮ?บิ?ฺ?฿?ๆใ?ช็?ํ๊?a{ฟXช?ฒx[ฟr?๒y??Cf?ช=Svฟึ}x?หZฟแ<?*;?g1ฟฐ}rฟถ}ฟทiฟgผ฿พรR?;๋`?7ฦ%ฟf,ฎพาe?้ฟื)w?M๚vฟฅซ??nฟ?๊?>@+hฟฆ{I?็จ>v}ฟ2ฦฝจค~?A|>maฟ=ฟvๆ>ึฅ{?F0?^'พนึZฟุฌwฟำ
ฟn6?==2?่z?'ฅo?*#?SR>wพๅฟVx^ฟน๓|ฟด{ฟ%์`ฟ54ฟB๙พ๙พoผsL>ึษ>"q?ห2?j&N?tใb?mq?Jz??๒??i}?็วx?#r?j?(a?VW?B8M?อ๊B?h8?๘e.?Oj$?น?a?ชk?ป?>6v๏>W	เ>นsั>กฑร>zฝถ>Zช>l">@k>b>??>&lp>&`>ูฏP>ชaB>ฆ5>(>ษ๙>'>แ>?X?=ึ๋=?=>Wฬ=@2พ=ฑ=;รค=X=ท=ฒา=฿9w=f=V=DG=Wq9=c,=ผ =t=O=)n=าไ๐<d,เ<ํะ<bฃAพ<N?์ษฟตfKฟYgdพ\ป฿พสุฟmj>p/v>sSฝg7-ฟU?ึเT?ค>v>rิ>Cf?7O?๔พCฟeปp?าไพ#ึผ2\>ตบพ๕ภO=5*ธ>?Sฟฝv{?ุทืพ
ํฟ?บq?w๋>K|ฟcาฝax?/ฅ๒>ฆ,ฟ2sฟ:<พO\9?ด|?5ึ?ยพ Wฟ*~ฟย7ฟl5Kพชด>`CE?ฑz?/ox?๖ยI?R?>U>?ว:พฟ๔พ๏5ฟ?ช_ฟพwฟZ๔ฟ6?zฟPkฟ้	TฟF7ฟศฟ0!ํพ^Iชพย่Qพo๊ฉฝ =|>wq>0ฆ>ฝ8ฯ> ๔>1q
?๊	?2๖%? ^1?h;?19D?๓K?พถR?oX?ว]?{Fb?่0f?Si?l?#o?`q?&Rs?u?เxv?	พw?ุx?ฬy???z?iX{?ต๗{?ม|?a๙|?a}?ิบ}?ก~?L~?g~?๙ธ~?ฦไ~?ถ
?+?H?ฎ`?v???dฆ?fฒ?อผ?ฮล?อ?\ิ?5ฺ?F฿?ฉใ?u็?ฟ๊?0ฟฤ??BrฟZพฑ?ั>8จฟ#ฟi
s?Bfฟซ>มt\?หi*พญ@ฟ๓u^ฟ8]8ฟวู
พฃ๖K?ฃy7?QฟฌลเฝjF?oๆ{ฟAT?ฮ?~ฟQ~?A๊]ฟ฿>%๙ก>ใปsฟะ๑5?ao?>ศวxฟxพi{?~ฉฅ>๔ณWฟHฟ >Yx??ต:?Lไฝ๙TฟฑZzฟG?ฟ?Jฎ=#i+?ฉ	y?:_r?Tๆ(?ฐm> ต]พีฝฟ?[ฟ|ฟ5|ฟcฟ7ฟํ?พฯพาฝฎ?>"-ฤ>ต'?มจ0?บL?oฺa?vปp?ฟ)z?๖~?๋?๒ภ}?ฬy?r?๔j?a?๋W?ัM?C?n39?u?.?๒?$?kH?๊๊?๏??Z ??c๐>u๊เ>SHา>?yฤ>๘yท>hAซ>ศ>ไ>ว๓>E>๏jq>)๑`>Q>1C>ส5>๖I)>?ก>_ฤ>๛ฃ>?h?=!ิ์=สs?=:3อ=?พ=Hลฑ=ัtฅ=ี?=Q=๗a=Dx=ูg=.W=H=9:=ฐM-=G!=๙=|ซ=๋๙=๔่๑<yแ<6~ั<ฬ_9? ษ;ฃ>?|ฟF?Sฟขฌiฟ QฟHRE?ึ?พฐภ฿>?Iqฟ"?n|?๙(?๑Z?>1?ธข}?ต?2ฟ?ฟt~?ฏร!ฟฃ6>ํ+=H้ดฝFx๊ฝ%C?>นOfฟ'ฺr?ํพ฿4ฟๆKg?๘xq>l฿|ฟ็?@พฑ:r?ั?	?ฟ๕?wฟูvxพป$/?yf~?๐T?g?Uพ\Qฟ?ฟ$>ฟฝ0mพ4ำค>`@?y?ํy?้pM??I2>&'พx์พ3๎2ฟJน]ฟหvฟ?ฟI{{ฟvlฟ`UฟใF9ฟํฑฟ?๑พ,ฎพJkYพ`'ธฝJ๙ฮ<๗Z>v?k>9งฃ>s๐ฬ>T?๑>j	?t;?ฒ?%?&ฝ0?ชฺ:?ฝC?็K?บWR?vLX?6]?jb??๙e?ii?fl??n?<Aq?๎6s?q๊t?adv?Aฌw?ขศx?<ฟy?	z?^N{? ๏{?6z|?ื๒|?][}?์ต}?`~?WH~?6~?5ถ~?aโ~?ค?ว)?zF?U_??t??ง?ขฅ?พฑ?;ผ?Pล?.อ??ำ?ใู????kใ?@็?๊?_r>\/?ฑฝฝพ?Qฟๆพ์พ9tฟวฐฝฺผ"?QฟฏIพใ-?ะๆ>e#ใพฯ"ฟp=ๅพN7>tp?ล[ ?Pแoฟu?>โ1?pMmฟ?2~?j๚~ฟผv?^ถGฟตw>๔แ>่{ฟ๛?cไ?ยqฟZ?|พ/Lv?9ฬ>ฤLฟช2Sฟ็?>#%t?SBD?Bจsฝ{โLฟ๓|ฟำฑฟ=4<=ปZ$?แv?C?t?ฆ|.?so>)DพวำฟCชXฟ {ฟBN}ฟ eฟ(:ฟบ\ฟ๊พpf:ฝ\3>ปณพ>ฺ
?1ฬ.?$JK?อ`?ฮp?ฟy?ฯห~?s๔?๔}?ญqy?ห?r??k?โ%b?X?jN?ฤD?้ฬ9?/?9%?๓ึ?t?Hr	?
ุ ?Q๑>]หแ>ยำ>8Bล>Y6ธ>_๒ซ>n?>yข>o>>จir>฿a>AlR>i D>6>]?)>0J>%a>6	>y?=ธัํ=?_?=3ฮ=๐หฟ=ฒ=e&ฆ=(ฃ=g๋=;๑=NOy=h=AํW=ช๒H=ฎ;=?.=๔!=Vท=ฉA=ฎ=ํ๒<โ<_า<ฬนx?%\Lฟซลm?ุ ฟ๙|ฟ<vฟPพf?๎Eฟ๙S?ฬC}ฟT?ฃ=มw?
ne?กฤE?่d?ฐฺ{??ฬฑ>ย~]ฟ%ะฒพึ๎}?EIฟภ>็}๒ฝ๗ฬถ=Dพย( ?ฎ้sฟฅํe?>พฺGฟำ1Z?iฮฉ>^xฟ๎พฦj?จค?Rญฟi{ฟข พg_$?๑?ญ|?แH(พQ6Kฟศบฟ=FDฟ๙rพj>๘R;?จIw?FB{?IQ?wZ?๔F>ง,พH[ไพ\เ/ฟบ[ฟ|หuฟผฟr|ฟpmฟWฟม;ฟ๎ฟ้๓๔พิฒพฐ๊`พbฦฝส<8>Bf>ผก>๊ฆส>B๑๏>?dl?ข$?ฦ0?รL:?@C?ฺK?\๘Q?-๙W?6]?ศa?เยe?9i?ฺ<l?H?n?ย!q?s?นาt?ฬOv?gw?'นx?ฯฑy?ez?GD{?Bๆ{?ขr|?G์|?ญU}??ฐ}? ~?คD~?~?oณ~?๚฿~??๙'?๊D?๛]?ฐs?}?ฦ?เค?ฑ?ฉป?าฤ?ภฬ?ำ?ู?ธ??.ใ?็?c๊?ฒr?ซพ(L๏>_0{ฟธ#IฟฐแSฟ๋qฟิt?>B>)ฝฟ/dn?n6 ?๎ฝkขพ"?ฝd?๓>?ด~>ะซ~ฟ|lถ>?>ฅปTฟะs?M๊vฟซ?f?yศ,ฟj(>ฝ ?ฟปใ?ฆ??ำgฟ?ซพ@Vo?ฬA๑>jฎ@ฟล\ฟ฿๚=o?๕/M?1ฉ๋ปข+Eฟ{*~ฟ$ฟUH?;:?ot?w?J์3?tํ>q}*พsิฟUฟzฟ?๗}ฟgฟฌ็<ฟ๊บฟ~&พG๒pฝ๛&>า3น>g?(๋,?;ีI?jป_?*Lo??Qy?์~?๛?ๅ$~?รy??is?k?wฐb?Y?ิO??นD?ืe:?+0?!$&?.e????K๕	?ํT?ฤ>๒>ฌโ>๑ำ>N
ฦ>๒ธ>?ฃฌ>|ก>?=>
>ผ>Qhs>อb>fJS>ผฯD>๛L7>ฟฒ*>^๒>ๆ?>&ศ	>งD >Lฯ๎=-L?=*๋ฮ=ลภ=ฝBณ=๘ืฆ=yH=M=~=Zz=]?h=SิX=ปษI=ูษ;=Hย.=ไก"=ณX=ีื=p=6๑๓<คใ<ส@ำ<ฯฬฆ>ๆบuฟQb?E>๘\ฟฌฟฌ>n^?ส๓}ฟV?\NฟwบพfซG?๘k?วr?|~?ผa?&-w=L๔wฟฏfะฝ3o?gฟi?ํพั/>
ธ?พๅ<?|ฟ๗๋T?+ฝฮ๔XฟVJ??-ู>ลqฟณตพNฬa?k(?Nฟt~ฟ๊5ทพ
?N??BG#?ฮต๔ฝฦwDฟ?ฟ?$Jฟ%พฆ?>์6?ฦau?Gm|?sT???[>Mษ พ?+?พ0ฦ,ฟญYฟพtฟฟฌ|ฟBชnฟSXฟึล<ฟ}zฟAึ๘พฆ่ตพูfhพ0ิฝ{2I<(๖=ฑค`>>&\ศ>๎ใํ>๛ด?บ?ั#?แy/?cพ9?จรB?hฌJ?ฅQ?ฅW?ฃํ\?a?e?s	i??l?ๆถn?(q?) s?่บt?";v?zw?ฉx?Tคy?ถ}z?&:{?z?{?k|?ฏๅ|?๗O}?ฌ}?า๛}?ํ@~?ห|~?ฆฐ~??~?x?*&?YC?\?r?y?ไ?ค?lฐ?ป?Sฤ?Rฬ??ำ??ู?p??๐โ?ีๆ?5๊?@๓H?๒oฟฉ้y?3P
ฟสฟซ~ฟวฟฟชf?=ุพฎV฿>?sfฟ0{-?b็c?|ฃ>B-=1แZ>Eใ9?y?ท	ธผ{}ฟ.ฺ?ฯ>d>_93ฟห`?gฟ๗งQ?ผรฟณ<๐,)?9ฟ๙?>ฃX1?9ิ[ฟโุพ#f?รi
?3ฟUEeฟ?โl=i?wU?า8=?<ฟkJฟzB,ฟ-ฝ?ฤตq?ธy?49?rO>ตพภ	ฟ*pRฟ?xฟ@~ฟุ
iฟ๒ผ?ฟํ
ฟ-&พนฝ	๘>ญณ>ฉ2?ฑ+?ุ[H?sฅ^?n?เx??l~?็??NS~?_z?8ำs?ยl??9c?คY?บO?RE?7?:?<ม0?ซถ&?๓?{?x
?งั?ร+๓>ใ>ลิ>Aาฦ>ฦฎน>Tญ>\บก>tู>จ>่>๋ft>ไบc>(T>E>_8>g+>>ค>8Z
>ฟฬ >?ฬ๏=Z8฿=วฯ=eม=uด=ง=สํ=3=ม=นd{=๕i=dปY=ฬ?J=<=|/=FO#=๚=n=2=V๕๔<น๔ใ<"ิ<tฟาsฒพ9๚]>QkW?ฟฃฉ=้uะ=RV???>5hฟK\f?ั๔?พA<ฟะ3้>0r?/ล?]z?,0?oSmพv๏ฟ๔>่S?Cyฟฬ6?ฏ?๕พเh?>็ฟ๎)U?C๐ฟj"@?3{=2+gฟ8?1?hฟk?พๆZW?ฅ6?วใพOฟ๚ฟำพษN?@ฝ?ผฎ,?;Yฝฮ_=ฟZ?ฟzพOฟ ซจพ๗Tl>ผ0?ศJs?Rn}?IวW?ชว?ิp>ำณฺฝแ๊ำพ็)ฟๅWฟญคsฟJUฟ๐}ฟ๎ทoฟธZฟ~>ฟYฟำณ?พrยนพฉ฿oพถฯโฝ4ฝ;ฒ?้=[>c>*ฦ>Yี๋>Vส?yฬ?า#?wื.?/9?HFB?>J?8Q?ญQW?pค\?ยHa??Se?ูh?๕่k?_n?mโp?ไr??ขt?b&v?zvw??x?หy?๚qz?๚/{?ชิ{?cc|?฿|?;J}?ง}?๗}?3=~?y~?ูญ~?$?~?^?X$?ฦA?B[?Uq?s??Xฃ?ยฏ?บ?ำร?ไห?เา?์ุ?)??ฒโ?ๆ?๊?	วฝๅVฟํ0L?๑~n><,ฟe2ฟแท=็|??'Wฟ|ุS?ิฟ>ช?ฎW(?1๏ส>ต๒?ฅhg?ฑ]?	พทkฟb>?rฒ;/
ฟ)E?.ํOฟฺ6?*ศึพ?๗ฝ8A?่{ฟ<ฅ>ฺD?+ขMฟฮ ฟฆ\?ฅL?lU%ฟiงlฟฝ็ปหBb?T]?iLว=ใ[4ฟ,ไฟ4ฟHืฝ๋่?ดn?ีz?ภR>??ฌ>ฉํฝีฟ+Oฟ<wฟฟมใjฟวBฟ^ฟคพJ๓ฎฝe>8!ฎ>๔ุ?ู)??F?6]?๔ฬm?kx??8~????W~?0az??:t?l?ยc?5Z?อ.P?/๊E?;?ํV1?ึH'?ฝ?ุ?น๚
?8N?~๔>ฯlไ>ี>ว>ัjบ>ธฎ>)`ข>?t>:>
ง>teu>ณจd>U>InF>ฝฯ8>s,>ซB >_7>G์
>ีT>hส๐=$เ=ฃะ=j2ย=+ภด=;จ==น==mo|=?ํj=tขZ=?wK=.Z==฿60=ง?#=l=.=๔(=w๙๕<ฮๆไ<\ี<ษ~ฟ ?hฟถ๕x?ฆe=?E7?-\?ียพน
ฟบ?!=ั4uฟfงQ=_?@?ek?Y?๖ฺ>ดฟ๛ืtฟ?ห>L$+?r?ฟW?W#ฟ)W?n3ฟวbh?ว~ฟ}๏'?ฒํ<>KMrฟ$?~{?]ฟmผฟฎK??sC?ณ@รพ\?ฟ%๏พ+?๓ศ~?fญ5?Pฒ๎ผค๑5ฟTWฟ/Uฟ??ธพ$ถL>้5+?q?<E~?๎๛Z?Lc??5>ภณฝหพปm&ฟmUฟ~rฟqฟ;}ฟkฝpฟชo[ฟt1@ฟ5!ฟEF ฟ(ฝพUwพg๑ฝึd9บ?=ฆdU>v>๛ยร>ล้>฿?ก๛?`"?4.?4?8?yศA?ZะI?+ุP?u?V?๖Z\?ธa?8e?จh?รพk?ฒmn?ยp?๓ศr??t?v?hdw?Rx?4y?1fz?ฤ%{?ะห{?ท[|?kุ|?xD}?ข}?1๓}?t9~?Qv~?
ซ~?ตุ~?C ?"?1@?ไY?%p?l??ข?ฏ?๏น?Sร?uห?า?ุ?แ??sโ?iๆ?ื้?งืcฟฒbฝ]ใj=ไ\?FC<ธฝา(?u9??ฟTR?เบ]ฟnๆ$พhกn?mMe?ฉต2?๙ฝG?'๛}?51.?ฎ-
ฟ!ึIฟCa?๋\พฌถพc"?๚:2ฟ?๓พI๑พ?ิU?๛าsฟ'X>ีฃT?!b=ฟฟใ๗O?-+?฿9ฟโrฟYฝZ?6๚c?๋ั>0M+ฟn๗ฟฃ;ฟ.?โฝJ	?นlk?2S|?CGC?$ตน>ภนฝผ?พ&อKฟข1vฟeฟฆฉlฟ๚>Eฟฃฟfซพ	%สฝpฬ >่จ>Y{?ฎ-'?ว[E?นl\?jm?๓w?9~?M???จ~?๙ฌz?อ?t?อm?Id?ลZ?ฤP?คF?P-<?)์1?ฃฺ'??๗?#}?ส?๗๕>?Lๅ>ลlึ>ปaศ>ภ&ป>Rตฎ>ไฃ>3>ห>#/>ํcv>ve>ไU>=G>9>ลฯ,>ห๊ >ิ>R~>่?>๑ว๑=ซแ=?~ั=9?ย=เ~ต=ช์จ=h8=?R=D.= z}=ๆk=[=๋NL=X">=*๑0=	ช$=ศ<=Z=ตด=?๖<โุๅ<ฅไี<Dq้พยr?'ฟ^d?X๛?ฃP??q@?w[0ฟฉa)ผช?=ข็?>ซจ|ฟlaนพ๙ฆใ>หL7?E ?ฝY >!;ฟxWฟVy?ํ@๓>2zฟุ&o?ืฮEฟฤ7?Oฟชv?ฏwฟ0ม?เ>#5zฟ?์?,?Pฟ@HฟQZ>?IO?ฦกพ(Vฟ<4ฟ&๊่>"}?ฬ=>?ย}<ฉ0.ฟจl~ฟ*Zฟ่ษพ้โ,>]%?!n?โ๑~?^?/์?ฟY>ผฝ?5รพใ/#ฟR9SฟขJqฟ๛ฟ~ฟ๗}ฟฐบqฟ!ื\ฟ่฿Aฟ#ฟ+0ฟทlมพำฦ~พ2?ฝ*๖๋ปkBั=ยO>ซ็>tม>zด็>5๓?3*?ศฆ!?-?g8?=JA?พaI?iwP?๏จV?6\?oศ`?=ไd?เwh?fk?แHn?ขp?.ญr?แrt?ข?u?CRw?zx?{y?]Zz?{?์ย{?T|?พั|?ฐ>}?}?ฺ๎}?ฒ5~?s~?8จ~?Cึ~?%?~?ฐ ?>?X?๕n?d?9?ฬก?lฎ?[น?าย?ห?า?Eุ???5โ?3ๆ?จ้?@R]ฟ/ำB??*9ฟPv?6ณ0?ีถ?ิz?๑?Q>S๚Yฟ๘uf?ฟฎฟYษ3??f?gbh?๛ฐq?ภ]{?ฬเ>9O?ฟLฟะw?ลฬุพ"พ๔๓>ธ้ฟ?ภไ>ุ#?ฝ$หพ?ฅf?Khฟ3ญว=ฮณb?ญ=+ฟอ'ฟ"GB?๐๑9?)Fฟ๏wฟ ฬพ๘+R?)j?M>ึ!ฟ%ฟุฦBฟำEพำ๗๛>U฿g?ฝ}?}H?ฦณฦ>Bนฝย!๔พRUHฟvณtฟ2ฏฟb\nฟX๋Gฟ?฿ฟ4๒ฑพMๅฝ]่=ุ๖ข>ิ3?>:;%?*ีC? J[?๎=l?ww?งฦ}??๙?Fะ~?บ๖z?u?ช}m?ูฮd?"T[?~XQ?lG?ฤ<?๐2?l(??ึ?Y???F?+๑๕>ต,ๆ>W@ื>B)ษ>โป>ิeฏ>ซฃ>|ซ>?\>1ท>Vbw>+f>ยV>ดH>cR:>->ๆ!>ษp>[>๚d>vล๒=ฯ?แ=์Zา=ฬร==ถ=8ฉ=ถ?=฿์=ฝ=ั~=Z?l=p\=๚%M=๊>=uซ1=jW%=$?=0=w@=ท๘<๗สๆ<๎ลึ<tจ ?์&??ล0ฟfพ้<9?VจO?นภL>.zฟซ>??์?พ:ึX?ํQฟk>6ฟ?=หษึ>ฦจ>๓Aพฎfฟ*ฟDK?ว>ป๊gฟoภ|?aฟEeT?Aeฟr~?โ?jฟ &?>.ื>ฤว~ฟ2ืํ>ฤK>??UAฟฌต&ฟญ๔/?ถ๕Y?wพf}ฟ้)ฟu๋ฮ>หษz?บZF?ด{=e &ฟณ}ฟฑฺ^ฟ?ุพnใ>ธ?`๏k?&t?4a?มa?|n>ศVKฝOรบพๆฟผ๘Pฟ
pฟํc~ฟฯX~ฟตฏrฟ8^ฟeCฟfแ$ฟฟ=ลพ{พLฏพ^`ผส๑ฤ=ฑJ>ถW>%ฟ>5ขๅ>ผ?.X?๏์ ?!ํ,?"7?ห@?ฟ๒H?NP?TV?0ว[?้`?ฌd?๙Fh?฿ik?๋#n?zp?Lr?ญZt?ข็u?@w?ษjx??my?|Nz?7{? บ{?HL|?ห|?แ8}?}?~๊}?๋1~?สo~?cฅ~?ฯำ~??~?ุ?=?"W?ยm?[?S?ก?ภญ?ฦธ?Qย?ส?พั?๑ื?P??๖แ??ๅ?y้?<5ฝพy?~ฟU๐>??&x?บf?eSฦพฌแพิ๊?Xฝ๑R`ฟโฒ>ฌr?พi???าา_?}๔">	/fฟฑถฦพะ๑?้ฟ?;=ฦ> -ฮพ,>sA๎<ฯฟK^s?0Yฟ7ผ<&n?5cฟ^
9ฟง!3?mG?z"๋พฆฦ{ฟTMพ?๕H?ro?แ>??ฟ~ฟdIฟ%เ@พ๋>fd?`~?^ญL?Dำ>]7#ฝSd้พฤDฟ?sฟiแฟา๛oฟณJฟ(ฟจฮธพุ5 พฯ=7Y>si๙>D#?6JB?#Z?pk?q๗v?๋}?ฅ๒?*๕~?t>{?gu?๔m?qSe?)โ[?์Q?ฎG?/Z=?C3??(?ฯ'?vฅ?\?๐ย??๖>V็>ผุ>ฆ๐ษ>Eผ>?ฐ>Qค>ตF>[๎>6?>ฏ`x>ำqg>?W>??H>ญ;>W8.>?:">y>aข>	ํ>๖ย๓=๐่โ=ื6ำ=าฤ=E?ถ=ฤOช==ย=ลL==ึm=ขW]=	?M=ชฒ?=ฟe2=ห&==ฑฦ=8ฬ=ื๙<ฝ็<6งื<๊ฟ?ๅ$dพิ=X?aฟ4ใq=0ฉ{>?ฯ?พฅlฟสe?cทSฟ+O?6ฒ๖พโ?oฟL	ฃพ=าป ๘พ?ผ|ฟเพร๎k?sช<ะJฟ	ป?ค$tฟSj?ฝtฟGไ??Zฟุ>?๕ฟาึป>ฺpN?$่0ฟBใ6ฟฏj ?.hc?@9พ๗ฬzฟ?ฟMEด>ฝมw?D?M?Rฺ=ฤฟ๙j{ฟ'Ncฟ๘่พรู=zร?U i?๓ห?๘ฺc?vร#?๛r?>ัB๚ผคAฒพฟwซNฟ?ฝnฟI?}ฟฐ~ฟpsฟl_ฟเ-Eฟฒ&ฟm?ฟ
ษพฉฯพ฿รพึ_ฅผQธ=ฝwD>ทฦ>Zิผ>ปใ>ฉ??2 ?ฉH,?d๏6?{L@?^H??ดO?๕?U?ใ|[?%G`?ฃsd?โh?.?k?ะ?m?>bp?Nur?aBt?าu?ภ-w?๋Zx?`y?Bz?เ{?
ฑ{?D|?Pฤ|?3}??}?ๆ}?!.~?l~?ข~?Xั~?โ๙~???i;?ฟU?l?Q?m?>??ญ?0ธ?ะม?%ส?]ั?ื???ทแ?ฦๅ?J้?/Q?฿ฏฮ>ฃฦฟกพลNE?Bf?'g้>_ณWฟลtุ=บี=ฦนอ>vฏฟัฝ?A?8ยt?ฆq?.?E	พ~็{ฟYฒพ|z?์Eฟฃ~>เ้=?pพ๛>ุm:>ฉ?#ฟsฤ{?ฟFฟ*พPูv?ฟธญHฟค"?nลS?Kfศพud~ฟม?พ1??cKt?u>zศฟ.}ฟ,Pฟq.hพๅลฺ>๘_?ฺH?เQ?&<เ>UkผL?พ1Aฟ	hqฟต๛ฟีqฟ?MฟF=ฟdกฟพฟพfฬต=7ถ>ฒ๔>ฐI!?๓บ@?๓๗X?3j?Dtv?H}?ฌ่?ญ?%{?Pศu?in?ีึe?0o\?โ~R?๑CH?ศ๏=? ฉ3?ห)?9ด?ึ,?,?ู>?ศศ๗>ฟ๋็>๕ๆุ>ไทส>?Yฝ>ฦฐ>?๖ค>?แ>ญ>0ว>๘^y>n_h>e~X>?ชI>๏ิ;>์.>ใ">%ช>d4>u>tภ๔=ีใ=ฟิ=eล=๕บท=Pซ=N(=ฃ =?=M=ำฮn=ฏ>^=ิN=าz@=	 3=+ฒ&=? =?\=๚W=๖	๚<ฏ่<ุ<อด?พ6jฟP!S?่Asฟส#ฟฆ{?พ>ฺcฟบ฿	ฟ๎~?Nฟใlj?ๆ KฝvฟV (ฟTพ$ฌพมฒ;ฟ|ฐ}ฟNp6พ\}?ฬ[พ฿c#ฟ๖w?FS~ฟื?x?ฤ}ฟปธ{?^ฐDฟ่o9>L[!?บ}ฟฐซ>ๅ\?f๒ฟ๘ฑEฟ)ึ?nk?ุ'ๅฝเ๐vฟฺo*ฟ?>	t?ว&U?,>ั ฟ$Uyฟtgฟซ๛๗พ? =พฌ?$f?;๙?๎f?ภ(?)fช>X;ผฑฉพฆ2ฟฅQLฟdmฟ}ฟH?~ฟ?tฟ.ๆ`ฟOอFฟ(ฟท?ฟืำฬพๆพฎึพฺผKฌ=2ะ>>ณ4>บ>zแ>?+?iฒ?w?ฎฃ+?/^6?๖ฬ??H?SO?ฉU?Q2[?"`?;d?ไg?Sk?ูm?แAp?4Yr??)t?aฝu?cw??Jx?JRy?6z??z?จ{?ธ<|?ฝ|?2-}?์}?นแ}?R*~?6i~?ฏ~?฿ฮ~?พ๗~?$?อ9?[T?Zk?E??u?eฌ?ท?Mม?ดษ?๛ะ?Hื?ฝ??wแ?ๅ?้?<Cm??0?พป
>bนfฟ?ื>aา>>U3พึฟ๛??พรH??nkฟE^๏พz*ไ>ำI?ห|F?ฉ)ึ>ฤุิพ6ฺ~ฟsใ= f?dฟิปแ>ฅาฝEฐpฝ๐Gฝ?Tฉ>D>ฟ)ฒ?๐>1ฟี!|พYณ|?yถึพVฟR๏?ฉ^?ภคพAลฟรฆพโ_4?5x?๑ณ>G=ฟฬ{ฟ>.Vฟพบษ>?[?๛ย?^U??ภ์>๒๛5<<ำพหY=ฟ$oฟ?ฟK sฟขOฟ^ฟjฦพำAพx=	>ถพ๏>ณJ?j'??ชศW?๚ษi?ํu?๙}?๐??ฬ7?อว{?_'v??n?Yf?7๛\?ีS?ญุH?ั>?<4?*?U@ ?๗ณ?ว?บ?0ด๘>๑ส่>บู>?~ห>Tพ>อvฑ>ฅ>๘|>๒> O>0]z>?Li>@\Y>zJ>*<>ิ?/>#>ฮF>dฦ> ?>ํฝ๕=)มไ=ฅ๎ิ=b2ฦ=ฃyธ=ฺฒซ=อ=บ=Dk=pา=วo=ผ%_=$ซO=๚BA=Rฺ3=_'=7ย=๓=ปใ=๛<2ก้<วiู<uFภพส^ฟ<w?ี?พ-ึ}ฟเciฟ*|ฟธ=x๐H?fฟR
?Nษ>ะLbฟพ,eฟฌ|ฟฝฌ!ฟWhฟิhฟ?ณม= l~?Lเพ่พฑฦe??ฟม?ท?ฟq?งE+ฟG=;ท8?\xฟธ฿#>?fh?ฟeSฟJฅ?>ดkr?สc-ฝK
rฟoฉ5ฟฅฒz>Iซo?้ฬ[?f?H>>9ฟ?vฟใJkฟฟถ0=Gu??b?๕๛?ณ!i?I,?๑Fด>๑(๛;ณกพqศฟl๋Iฟใ?kฟY
}ฟr@ฟํ\uฟI3bฟจgHฟXH*ฟeพ	ฟ(ะพ$4พข็พ~?ฝP๕='9>ญก>/ธ>4d฿>น=?ฉ??ผ?1?*?ฬ5?M??sฃG?๋๐N?ฟSU?y็Z?โฤ_?.d?$ณg?M้j?*ดm?d!p??<r?~t? จu?๓w? ;x?lDy?*z?๒z?{?ไ4|?ฦถ|?R'}?ี}?P?}?&~?ๆe~?ั~?cฬ~?๕~?G?08?๕R?$j?9~??ฌ?ทซ?ท?หภ?Cษ?ะ?๔ึ?t??8แ?Xๅ?๊่?ฉ/=>?}ฟZg?4หoฟ้ณฟ!ปพ๗<ฟ฿[MฟwSq?>SฟCต{?จ'ฟpำIฟ
'=G?j??_ฌ๋=2a)ฟ๔ฮnฟ 1ท>.WE??wฟY?๒sพๆ3๓=?1Jพ q๑>๙ฦTฟ?ฟ\gถพฃ?ธ<งพ๔bฟ%H?>ยh?ๆพU็ฟQำลพ )?๓V{?vฤฬ>ึฤ๐พz~xฟำๆ[ฟิพ]iธ>
W?ค๛?โoY?a๙>~ื=นhศพษ9ฟaดmฟ่ฟetฟ?Rฟ;u ฟI(อพฝ(พ=?`>ฃ?๊>ขG?ฅ=?<V??๐h?Gcu?ลผ|?pฬ?U?k	|?ตv?ฎNo??ูf?;]?๓กS?บlI?I??yฯ4?ฎ*?"ฬ ?ื:?.?-6?S๙>๋ฉ้>฿ฺ>๕Eฬ>ฐะพ>๐&ฒ>hAฆ>>*ข>ื>X[{>|:j>:Z>$IK>^W=>
U0>(3$>sใ>aX>'>bป๖=@ญๅ=สี=(?ฦ=P8น=bdฌ=โr?=dT=๚=วW=Iฟp=ศ`=1P=!B=4=๋(=c=2=|o=4?<F๊<Kฺ<ฯ{ฟSบ?ฝ*ฺ>Fณ>bPฟ\lvฟธก-ฟฺ?ฺช>ฟงธ:>vA?ษ|ฟ๑aฟX-[ฟนZฟขL~ฟZ๔?ฟ<vธ>o?#ฟ3~พ๔๎I?๗เvฟE0~?พ๕zฟจฟa?Xฟeญฝ้;M?4oฟUZ=b๕q?ี"๎พ๘ร^ฟ}๚ื>(็w?Xภเ<lฟ=9@ฟๆB>ขj?คํa?%v>ืฟtฟpัnฟZ๊
ฟ?๓<<t?ฉ_?"ิ?๊k?๔k0?Cพ>FO?<ำhพฝSฟ๑xGฟภjฟ|ฟyฟ0vฟดycฟ฿?Iฟิ,ฟnฟ]ิพWใพค๖"พ:r"ฝ=|3>ซ>i?ต>,M?>฿N ?W
? ?3X*?]:5?ฆฬ>?๋2G?oN?ฏ?T?ZZ?d_?"ษc?}g?พj??m?ว p?ซ r?่๘s?สu?p๖v?๒*x?6y?~z?็z?๑{?-|?๗ฏ|?k!}?ธ}?โุ}?ช"~?b~?๐~?ๅษ~?n๓~?h?6?Q?ํh?+}?ณ?โ?ซ?kถ?Gภ?าศ?7ะ?ึ?*??๘เ? ๅ?บ่?^':ฟ฿]1ฟ?i?ภศสพ8ืzฟฯ}SฟEา~ฟ?Hคพ?ตy?ฆIฟ฿Jt?lพ-xฟ~ขพ#5>ฐS>VLพฦYฟฬ๗Lฟ๗ๅ?7?!โฟb8C?ฮฤโพล>&eถพ6๋?ณ์fฟr๑y?Bแ?พฮO์พฦ?c.lพ[ตlฟคอิ>Kp?ฆ~4พส~ฟ?ไพ(?เฌ}?dๅ>?|ฺพsuฟP;aฟ|?ญพธูฆ>0R?ฦ๒?Q]?๚?~=\.ฝพจ5ฟ๕ณkฟปฟถuฟ\sTฟฑ#ฟป?ำพ16พ?~S=้ฎ>๗ๅ>@?ฌ๓;?ฑ]U?ใh?zีt?jr|?.บ?ไp??H|?Pเv?Nฟo?พYg?>^?<2T? J?2ญ??๔a5?=+?W!?vม?a?ฑ?2๚>ญ๊>_?>ฦอ>๎ฟ>๛ึฒ>ฐๆฆ>?ฒ>U3>โ^>pY|>๐'k>ึ[>+L>>>:	1>-?$>>\๊>->ิธ๗=Uๆ=hฆึ=๋หว=๛๖น=๊ญ=*ก=C๎=ฟ=?=ทq=ิ๓`==YQ=HำB=ไN5=Kบ(=ํ=]==๛=S?<Y๋<W,?<%ผ/ฟ8?sำพฐk?วLพb?ฟ๙Kฤฝดvr?aพยซฝพVw?ตStพ๓รrฟ๖{ฟฅzฟ๏ูzฟRฟ5`?M>P?U?Lฟ๐M๗ผ้%?neฟoอt?w4oฟ~L???พสm]พ^?
cฟ>โ]ฝy?	?ยพ*ืhฟึ่ฑ>ๅ?{?ฬฦ=์3eฟgJฟฝฏ	>a๕d?=g?บu>]๓พสpฟxrฟ$ฟgคผญฉ ?*\?ศ?7โm?าx4?อว>่฿;=ฑพฦิฟ[๚DฟJiฟU๊{ฟจฟ้๛vฟfนdฟ๊Kฟฯ-ฟษuฟSุพpพ*พ4=ฝSE=zะ->ฑx>3ณ>๚4?>฿พ?>t5?C?ดฑ)?ยง4??K>? ยF?+N?OงT?๖PZ?จA_?฿c?ฆOg?ฤj?๑hm?	เo?<r?9เs?^}u?ฺใv?ิx?(y?_z??z?ึ{?$%|? ฉ|?~}?~}?pิ}?ฯ~?=_~?~?dว~?C๑~??๑4?%P?ตg?|?ษ??Xช?ำต?ฤฟ?_ศ?ิฯ?Iึ?เ??ทเ?้ไ?่?txฟY(>ุ่>ุบฦ>yhZฟuฝ~ฟปสYฟcV>}05?-ฉfฟ๐4?ดจC>ิ{ฟ๏่'ฟธ>พJ?พอ?พ?qwฟWฺฟ|D?^+ส>ญ^{ฟ.ใ`?๊ฟ่๊>๘ZฟPy7?Vtฟ,]p??ิรพ6ฟฉ|?8พdพtฟ฿บซ>?mv?]๑ะฝ3p|ฟ๊จ ฟXช?~5?หซ?>ฐรพ็qฟH)fฟเภพ>ุM?dจ?#a?ฐ?:ณ=ฤุฑพๅ1ฟiฟขuฟ2๓vฟ๗ศVฟ:&ฟ
ฺพBCพ6ธ =[๘>ฯ	แ>x5?S:?"T?3g?(Dt?้$|?)ฅ?ฺ?|?1:w?{.p?Hุg?=^?ฏมT?ยJ?@@?๘๓5?ผฬ+?ฮโ!?ึG?_?ื,?หt๛>7g๋>2?>sำอ>Gภ>๎ณ>ใง>้M>tฤ>ดๆ>wW}>Vl>๕[>)็L>ฒู>>eฝ1>.%>ฒ>S|>0>Bถ๘=g็=Gื=ฌศ=ฅตบ=pวญ=rฝก=!=?=rb=ปฏr=฿ฺa=H0R=oC=,	6=ชg)=Gฆ=ต=?=q?<mw์<?<ฃลv>B|?2vฟใํk?,?A๐ส=ำ?2v?_ู4ฟa?>บ๓5ฟH{?8>ั=AฟD{ฟฤ}ฟว^ฟห@พ?K?้๛#?2kฟ?฿A>ธข๔>พฮKฟ	ฒc?๊\ฟ๓2?ฦพ๋1ฐพอl?๙Sฟaั$พxฟ}?J'พฅ+qฟฏ>	ช~?%8*>๛Q]ฟด4SฟJเ?=บง^?Ql?pง>ฌ(เพ1mฟใ่tฟช2ฟWธSฝร0๔>สX?๕?Ep?2o8?Mpั>N=ย๎พฬKฟฯoBฟgฟI{ฟอฟฤพwฟT๒eฟฟMฟQ/ฟmMฟุ?พa;พy1พ$Wฝมึu=#(>ฤโ>็/ฑ>กู>ึ??> `?d?ด
)?ฑ4?ฅส=?ดPF?oศM?ขPT?MZ?ฎ?^?fVc?g?@gj?Cm?+ฟo?ฑ็q?rวs??gu?2ัv?ฅ
x?}y?5z?าz?ฒ{?8|?Cข|?}?ny}?๙ฯ}?๑~?ใ[~?%~?เฤ~?๏~?ฅ?O3?ปN?{f?{???K?จฉ?:ต??ฟ?ํว?qฯ?๓ี???wเ?ฑไ?Z่?๓งคพฆc?qาฟgo?ำพ>ไ2ฟค4ธพT?E?uc>7Lฟ+>ฯ??ฐํRฟ?eฟv[ฟอ่พ}=ฟฮืฟ(?พพg?,>@ชjฟพu?๔อ>ฟฃp?๚#ฟสฏP?พ|ฟIb?pลพ'ฟษv?ฝ@ฉzฟก^>:6{?ใ?ผP?xฟ,ภฟaฅ?๏?ฯ	?Hmฌพ?mฟ|ฎjฟXำพค>วG??ฺd?.{?#็=iฆพ?l-ฟggฟ\ฟQxฟYฟง)ฟ? แพ Qพqึ?<หzv>X?>{&?Cฏ8?UโR?]Nf?Qฏs?Cิ{?b?n??ย|?Uw?2p?Uh?9!_?KPU?พ$K?Nำ@?6?[,?ญm"?๔อ?)?์ง?_?>E์>k?>๛ฮ>ม>ษ6ด>1จ>ฤ่>U>{n>mU~>ฎm>@ำ\>ถM>ั?>q2>*+&>Kน>G>1>ฌณ๙=uq่="^ุ=leษ=Mtป=๕xฎ=ธbข=?!=8จ=ว็=๓งs=้มb=SS=cD=tร6=	*=ขG=ฒK=พ	=?<iํ<็๎?<fr?y0๊>& Uฟธeท>v?U7??n?๏น"?พyฟuS?๕uฟL?ะ?้ญไพ่YฟdฟB",ฟn=อซm?JDฺ>ย\|ฟ,กฬ>๐ว>ฌ*ฟ1hK?DฟฝF?ฌึ"พ๎พฦีv?YBฟถ!พเ??ำ	PพoฑwฟหE>ฝ็?%6p>4Tฟ[ฟ{ท<พW?ะq?Kฝ>ภฬพ;iฟดwwฟ๏ ฟ$yชฝ
ุๆ>|ฏT?ฝ]~?ลr?N<?ํ?ฺ>9
ฌ=๚A|พนฟuู?ฟฐ์eฟrzฟf่ฟ+yxฟw$gฟUNฟ?G1ฟQ"ฟ฿พไพ 8พภ$rฝ ]=8t">้K>ุฎ>$ื>ง?๚>??ภศ?5c(?)3?I=?฿E?๋dM?ฆ๙S?]นY?wฝ^?ถc?h๋f?;j?$m?-o?	หq?ฎs?GRu?xพv?f๚w?gy??๙y?๘วz?z{?C|?^|?}?Bt}?}ห}?~?X~?;~?Zย~?ๆ์~?ภ?ซ1?OM?@e??y?๒??๗จ?กด?บพ?zว?ฯ?ี?K??6เ?yไ?)่?บ}?๛ๅe?ผ:~ฟซg?Al๋>rAฝ้(>฿?2๒ฉพมฝEพฉ๛c?mฟอ\ฟาPKฟRพ9ฟซiฟผ>rฟบpฺฝะ{?ีโฝBNฟ?ฟ[ฟH๘<?OBฟๅ๕d?0๗ฟผฅP?๏>พj?<ฟภn?cค=ิh~ฟ`,>]~?๘C=Qtฟa=ฟลJ์>~ฺ??็?dมพbTiฟ฿ศnฟ7ๅพีฺa>79B?pO~?๋ฬg?Q@?าd>kโพ|:)ฟ๓eฟ=ฃ~ฟ[1yฟุF[ฟสn,ฟูฑ็พ_Y^พgl<w?j>_ื>?ๅ7?Q?ฺee?๘s?z{?ุr?ด?v๛|?พ่w?sq?ตัh?2จ_??U?ถK?eA?7?ํ้,?<๘"?ัS?ฝ	?ึ"?,I?>ก#ํ>ึ?>]`ฯ>๕ผม>ๆด>ึจ>>ๆ>8๖>SS>๙๏m>ๅฐ]>
N>้[@>ฉ%3>!ำ&>โU>8?>/ฅ>ฑ๚=]้=๛9ู=)2ส=๓2ผ=x*ฏ=?ฃ=?ป=s7=m=*?t=๒จc=^?S=บ+E=ผ}7=hย*=?่=?แ=~	=W =[๎<.ะ?<?H?B<แพ฿s๐ฝฺูพUc?jC?*๏u?๕ฌ=w|qฟ;E?\6{ฟ{ไ่>ฒZ?qฝ5ฟ)+0ฟmWัพุฅ>'~?ฑ๏<>๒fฟแ2??ฒ=ฑSฟด,?
ไ&ฟSๅ>UผใTฟ๋V}?t}-ฟฎHผพาk?j๐ใฝ\|ฟฮ1็=7ต?h>หJฟZcฟ:
ฝ=P?๕t?=ฆา>ธธพ้dฟฒyฟไล&ฟy๊๊ฝ%Lู>ZดP?=}?kt?{@?้qไ>ำ=jพอฟv7=ฟทIdฟ`ไyฟ-๙ฟ+yฟฤOhฟ?Pฟ>?2ฟl๔ฟyDใพ?พ{?พ`XฝLhD=ฤ>Hh>ฌ>ๅิ>T๙>jณ?
?6ป'?,ํ2?๖ฦ<?๘lE?M?\ขS?)mY?{^?ะโb?นf?บj?๗l?}o?Fฎq?s?<u?ชซv?๊w?B?x?ปํy?Wฝz?Mq{?G|?s|?	}?o}??ฦ}?)~?%U~?N~?ัฟ~?ด๊~?ู?0?โK?d?๋x??ฑ?Eจ?ด?5พ?ว?ฉฮ?Fี? ??๕฿?Aไ?๙็?ลฌ~?<>;ฟ4ดฃ>?q??&?ฝSM?ฤ1^?
คHฟ0ฎ?>ป!ฟ๔?3?ฝ7?rฟ%vuฟืiฟ&~ฟาOฟA1*>โb?ำ?พhy(ฟ9~?ฑpฟ)X?@๖Zฟสะs?๑}ฟฟ;?zฅผุ๛NฟYb?ผy*>า๔ฟตBจ=L??ฬ๚=/nฟ๚)ฟ๓mะ>[๖~?ิ?kvyพ?Qdฟvrฟ ๗พJ==>๖q<?5A}?ๅj??่?0'>๙Dพเ๒$ฟ(ถbฟP~ฟ82zฟฮn]ฟuS/ฟจ6๎พจkพ๓c;v_>
า>๒??yZ5?ศVP?yd? {r?){?U?hฦ??2}?i=x?=sq?Li?%.`??jV?FL?#๗A?8ง7?๓w-?z#?mู???๔2?>๎>จ?>&ะ>ปwย>7ต>	{ฉ>K>w>๋}>(>7?n>^>๎SO>๚A>ยู3>{'>t๒>'2>+->tฎ๛=I๊=าฺ=ไ?ส=๑ผ=๚?ฏ=Aญฃ=ทU=ฎฦ=o๒=`u=๛d=hตT=฿๓E=88=ฦo+=V=x=>*
=e =ฅM๏<vฑ?<ใะฝฅ{ฟ-.?ึrฟ7@ฌ>P?	ๅ?ุI?พ?ฟถยf?^๓Fฟ-<3~?ล๒ก>%dพ]ะพะึฝ่|?~p|?Iบฝ%tฟ๊ฟ@?ป่๐ฝฒeฎพl?6ขฟ>>ม.ฟก๒?๔คฟC๎พom|?๓Pผq"ฟฦG=ฟ~?1-ผ>$:@ฟีำiฟ6ธฝA*H?Ix?ฌ็>ฌIคพd=`ฟ+{ฟF-ฟึพห>(L?|?๑ลu?rฦC?<ฮํ>+ไ๙=ฯXพHw ฟ๛:ฟพbฟํ yฟ??ฟิyฟ3tiฟQฟEฏ4ฟดรฟ๕ๆพฤ.คพs!Fพ์ฝ3ฎ+=ณ>๓6z>&ช>ษศา>?7๗>J??เK?ธ'?บX2?}D<?๚D??L?ฤJS?ฏ Y?P8^?ดจb?kf?ธใi?ฤะl?ะ[o?fq?|s?ู&u?สv?ธูw?๐x?kแy?ซฒz?h{?C|?|?}?ูi}?xย}??~?มQ~?^~?Fฝ~?่~?๑?^.?tJ?วb?ุw??ใ?ง?lณ?ฏฝ?ฦ?Eฮ?๏ิ?ดฺ?ด฿?ไ?ศ็?fl็>wธ-ฟส์1=์พgk?"๚w?'??พ4?>ห~ฟำSS?ฟykฟฌAh?นธ>๒mAฟ<ฟ2ฟ?Ozฟiฟ vฺ>Lฌr?;2ฟe๔พฝr?F฿|ฟฌGm?ฑัmฟxๆ|? ถvฟb-"?ดcิ=t๖^ฟบT?6>อIฟcbปeด?/I>_๑fฟ05ฟWฮณ>?C}?.ุ)?ฅาHพ?ึ^ฟ็ตuฟ4ฟ'_>s6?'๒{?7สm?"t?เ@>็พด ฟใ8`ฟฃq}ฟิ{ฟ?_ฟ|-2ฟ๑ฎ๔พํxพํ6*ผ!่S>อ>โ?ช3??
O?fc?ษ?q?ฯz?5?ะี?4h}?Wx??q?<ฦi?ณ`?๗V?ึL?2B?\78?.?i$?ศ^?G
?(?u?>)฿๎>Zz฿>ฒ์ะ>c2ร>ษEถ>๎ช>๖ธ>p>>vง>gสo>l_>ศ"P>?A>ึ4>#(>>ฤ>$ต>ำซ?=5๋=ฆ๑ฺ=หห=;ฐฝ={ฐ=Rค=๏=็U=รw=v=we=qU=ผF=J๒8=$,=ฏ+ =0=?ต
=t=ท?๐<ฝ฿<S[dฟ๗<ฟ$ย?2cฟjสพY~>ฬฒผื฿fฟยH?ฝ?|?ใ?ศพ7WืพRฯt?{ฑ'?c
=ี?ขฝ]ดV>ษVK?ใg?%ฃพ๎9[ฟห๓`??ฃขพ>พ0ภ>aฝพa?>รช>ผFฟข~?๛พdฟฟ_้v?พ=b?ฟฑDฝฐ{?|๋?>ฝู4ฟสฑoฟตGพ??V{?c?>ูฒพึ8[ฟg&}ฟ?3ฟ'5พๅฆฝ>ฑFH?๔j{?gw? ^G?ใ๗>๖W>๚Fพ๙พ,ั7ฟ?฿`ฟ$Rxฟw?ฟbuzฟผjฟ1SฟK]6ฟ!ฟษก๊พะงพ๔"Mพเ฿?ฝ๒=`>?u>ฬง>๐ชะ>HS๕>?ฅ?ผi&?ำร1?ม;?ทD?U8L??๒R?๏ำX?`๕]?anb?ฅSf?ทi?]ชl?q:o?jtq?^cs?u?ืv?Hษw?ฮแx?ีy?๔งz?ฤ^{?7?{?|??|?d}?๏ฝ}?Q~?ZN~?l~?ธบ~?Jๆ~??ต,?I?a?ฤv?'??฿ฆ?ัฒ?)ฝ?ฦ?เอ?ิ?iฺ?s฿?ะใ?็?โขฟย~ฟrJ?xดuฟ[ฮ>7?f?qJ?jง พภ fฟร@?๙ฟ?ฌ!?j(?D1ๅพZgฟฉ\wฟค4]ฟiปณพิล'?BIV?\?3ฟ??พ๑]?พุฟ[z?ฝ-zฟๅ??ljฟาf??pg>ย๘kฟ~๘C?ึท>?i|ฟ๏?นฝขโ}?ิ๋>ฦญ^ฟป@ฟฑ>hฤz?~ก3??ดพ๗ๆXฟexฟถณฟ ๆ=>0?bz?zp?ฉเ$?lpZ>วoพ&ฟjฃ]ฟEต|ฟ๗{ฟaฟฑ?4ฟ\๛พพภบผ๊RH>ไศ>\ฤ?๕1?<ปM?~b?๘8q?Srz?ฐ?ำโ?}?แx?kDr?ง>j??6a?LW?ตeM?ญC?	ว8?ึ.?$?แใ?<??ุ ?ผ๏>๘Kเ>ฅฒั>ํ์ร>B๕ถ>ฟฤช>S>N>0>O&>ทp>I`>๑P>B>ไA5>๋ส(>+>๚U>=>-ฉ?=!์=wอ?=Tฬ=?nพ=๛>ฑ=ฦ๗ค=l=!ๅ=?=ษw=
^f=ycV=(G=ฌ9=ส,=	อ =Yค=พA==ษ1๑<tเ<๎ฟ\ฟ๋ื=<?ฃผพ\jฟ;?พ๚ฑฟlิ|ฟิฮเ>นW=ฏpซ=|Fฟ้ธ@?'๋d?u3?>7แ>V? ?Wตo?L\A?๒ฟ	6ฟฑ>v?๘ฟ๎ภ=	mR>ผ!Yพย้:ปอ>แYฟ(`y?2ฦพgท$ฟฝํn?ิ?*>/๑~ฟ?Xพqv?{?>,ถ(ฟฝฏtฟ๛Mพดe6?ส,}?า๙?:ตuพ??Uฟ;_~ฟVฌ9ฟJUพซฏ>ษีC?z?ไx?ด?J?o ?Wฐ#>๓5พ๚"๒พ55ฟ&_ฟxwฟ๘๎ฟท{ฟUจkฟc~TฟF8ฟชYฟJ๎พ๖oซพๆ!Tพ!ฎฝุj๔<Jฌ>ฯo>pฅ>?ฮ>m๓>b,
?ไฬ?Bภ%?w.1?K>;?D?uำK?ซR?๊X?3ฒ]?ู3b?ฏ f?6i?ัl?๒o?RWq?Js?๛t?าrv?ศธw?ำx?งศy?3z?rU{?"๕{?|?s๗|?[_}?aน}?_~?๏J~?v~?(ธ~?ไ~?
?+?G?H`?ฏu?7?D?+ฆ?5ฒ?ขผ?ฉล?{อ?@ิ?ฺ?1฿?ใ?e็?

NoOpNoOp
น
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*๑
valueๆBโ Bฺ
ฑ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	mlp_units
		optimizer

positional_embedding
encoders
avg_pool

mlp_layers

mlp_output

signatures*

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18*

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18*
* 
ฐ
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
(trace_0
)trace_1
*trace_2
+trace_3* 
6
,trace_0
-trace_1
.trace_2
/trace_3* 
* 
* 
ภ

0beta_1

1beta_2
	2decay
3learning_rate
4itermmmmmmmmmmmmmmmm? mก!mข"mฃvคvฅvฆvงvจvฉvชvซvฌvญvฎvฏvฐvฑvฒvณ vด!vต"vถ*

5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;	embedding*

<0*

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 

C0
D1*
ฆ
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

!kernel
"bias*

Kserving_default* 
xr
VARIABLE_VALUE8transformer_model_17/positional_embedding/dense_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_model_17/positional_embedding/dense_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJtransformer_model_17/transformer_encoder/multi_head_attention/query_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEHtransformer_model_17/transformer_encoder/multi_head_attention/key_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEJtransformer_model_17/transformer_encoder/multi_head_attention/value_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEOtransformer_model_17/transformer_encoder/multi_head_attention/projection_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEMtransformer_model_17/transformer_encoder/multi_head_attention/projection_bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEBtransformer_model_17/transformer_encoder/layer_normalization/gamma&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAtransformer_model_17/transformer_encoder/layer_normalization/beta&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_model_17/transformer_encoder/conv1d/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4transformer_model_17/transformer_encoder/conv1d/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_model_17/transformer_encoder/conv1d_1/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_model_17/transformer_encoder/conv1d_1/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEDtransformer_model_17/transformer_encoder/layer_normalization_1/gamma'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUECtransformer_model_17/transformer_encoder/layer_normalization_1/beta'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_model_17/dense/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEtransformer_model_17/dense/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_model_17/dense_1/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_model_17/dense_1/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
* 
.

0
<1
2
C3
D4
5*

L0
M1
N2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
ฆ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

kernel
bias*

\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b
attn_heads
c
attn_multi
dattn_dropout
e	attn_norm
fff_conv1
g
ff_dropout
hff_conv2
iff_norm*
* 
* 
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

otrace_0* 

ptrace_0* 
ฆ
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

kernel
 bias*
ฅ
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
}_random_generator* 

!0
"1*

!0
"1*
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
M
	variables
	keras_api

total

count

_fn_kwargs*
* 

;0*
* 
* 
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
* 
* 
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
?trace_1* 
* 

ก	variables
ขtrainable_variables
ฃregularization_losses
ค	keras_api
ฅ__call__
+ฆ&call_and_return_all_conditional_losses
งdropout
query_kernel

key_kernel
value_kernel
projection_kernel
projection_bias*
ฌ
จ	variables
ฉtrainable_variables
ชregularization_losses
ซ	keras_api
ฌ__call__
+ญ&call_and_return_all_conditional_losses
ฎ_random_generator* 
ถ
ฏ	variables
ฐtrainable_variables
ฑregularization_losses
ฒ	keras_api
ณ__call__
+ด&call_and_return_all_conditional_losses
	ตaxis
	gamma
beta*
ฯ
ถ	variables
ทtrainable_variables
ธregularization_losses
น	keras_api
บ__call__
+ป&call_and_return_all_conditional_losses

kernel
bias
!ผ_jit_compiled_convolution_op*
ฌ
ฝ	variables
พtrainable_variables
ฟregularization_losses
ภ	keras_api
ม__call__
+ย&call_and_return_all_conditional_losses
ร_random_generator* 
ฯ
ฤ	variables
ลtrainable_variables
ฦregularization_losses
ว	keras_api
ศ__call__
+ษ&call_and_return_all_conditional_losses

kernel
bias
!ส_jit_compiled_convolution_op*
ถ
ห	variables
ฬtrainable_variables
อregularization_losses
ฮ	keras_api
ฯ__call__
+ะ&call_and_return_all_conditional_losses
	ัaxis
	gamma
beta*
* 
* 
* 
* 
* 
* 
* 

0
 1*

0
 1*
* 

าnon_trainable_variables
ำlayers
ิmetrics
 ีlayer_regularization_losses
ึlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

ืtrace_0* 

ุtrace_0* 
* 
* 
* 

ูnon_trainable_variables
ฺlayers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

?trace_0
฿trace_1* 

เtrace_0
แtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
5
c0
d1
e2
f3
g4
h5
i6*
* 
* 
* 
* 
* 
* 
* 
'
0
1
2
3
4*
'
0
1
2
3
4*
* 

โnon_trainable_variables
ใlayers
ไmetrics
 ๅlayer_regularization_losses
ๆlayer_metrics
ก	variables
ขtrainable_variables
ฃregularization_losses
ฅ__call__
+ฆ&call_and_return_all_conditional_losses
'ฆ"call_and_return_conditional_losses*
* 
* 
ฌ
็	variables
่trainable_variables
้regularization_losses
๊	keras_api
๋__call__
+์&call_and_return_all_conditional_losses
ํ_random_generator* 
* 
* 
* 

๎non_trainable_variables
๏layers
๐metrics
 ๑layer_regularization_losses
๒layer_metrics
จ	variables
ฉtrainable_variables
ชregularization_losses
ฌ__call__
+ญ&call_and_return_all_conditional_losses
'ญ"call_and_return_conditional_losses* 
* 
* 
* 

0
1*

0
1*
* 

๓non_trainable_variables
๔layers
๕metrics
 ๖layer_regularization_losses
๗layer_metrics
ฏ	variables
ฐtrainable_variables
ฑregularization_losses
ณ__call__
+ด&call_and_return_all_conditional_losses
'ด"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

๘non_trainable_variables
๙layers
๚metrics
 ๛layer_regularization_losses
?layer_metrics
ถ	variables
ทtrainable_variables
ธregularization_losses
บ__call__
+ป&call_and_return_all_conditional_losses
'ป"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

?non_trainable_variables
?layers
?metrics
 layer_regularization_losses
layer_metrics
ฝ	variables
พtrainable_variables
ฟregularization_losses
ม__call__
+ย&call_and_return_all_conditional_losses
'ย"call_and_return_conditional_losses* 
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ฤ	variables
ลtrainable_variables
ฦregularization_losses
ศ__call__
+ษ&call_and_return_all_conditional_losses
'ษ"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ห	variables
ฬtrainable_variables
อregularization_losses
ฯ__call__
+ะ&call_and_return_all_conditional_losses
'ะ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


ง0* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
็	variables
่trainable_variables
้regularization_losses
๋__call__
+์&call_and_return_all_conditional_losses
'์"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

VARIABLE_VALUE?Adam/transformer_model_17/positional_embedding/dense_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/transformer_model_17/positional_embedding/dense_2/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ฎง
VARIABLE_VALUEQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ฌฅ
VARIABLE_VALUEOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ฎง
VARIABLE_VALUEQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ณฌ
VARIABLE_VALUEVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ฑช
VARIABLE_VALUETAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ฆ
VARIABLE_VALUEIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ฅ
VARIABLE_VALUEHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE;Adam/transformer_model_17/transformer_encoder/conv1d/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ฉข
VARIABLE_VALUEKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
จก
VARIABLE_VALUEJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/transformer_model_17/dense/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/transformer_model_17/dense/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/transformer_model_17/dense_1/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/transformer_model_17/dense_1/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE?Adam/transformer_model_17/positional_embedding/dense_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/transformer_model_17/positional_embedding/dense_2/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ฎง
VARIABLE_VALUEQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ฌฅ
VARIABLE_VALUEOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ฎง
VARIABLE_VALUEQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ณฌ
VARIABLE_VALUEVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ฑช
VARIABLE_VALUETAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ฆ
VARIABLE_VALUEIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ฅ
VARIABLE_VALUEHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE;Adam/transformer_model_17/transformer_encoder/conv1d/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ฉข
VARIABLE_VALUEKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
จก
VARIABLE_VALUEJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/transformer_model_17/dense/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/transformer_model_17/dense/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/transformer_model_17/dense_1/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/transformer_model_17/dense_1/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
ี
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_18transformer_model_17/positional_embedding/dense_2/kernel6transformer_model_17/positional_embedding/dense_2/biasConstJtransformer_model_17/transformer_encoder/multi_head_attention/query_kernelHtransformer_model_17/transformer_encoder/multi_head_attention/key_kernelJtransformer_model_17/transformer_encoder/multi_head_attention/value_kernelOtransformer_model_17/transformer_encoder/multi_head_attention/projection_kernelMtransformer_model_17/transformer_encoder/multi_head_attention/projection_biasBtransformer_model_17/transformer_encoder/layer_normalization/gammaAtransformer_model_17/transformer_encoder/layer_normalization/beta6transformer_model_17/transformer_encoder/conv1d/kernel4transformer_model_17/transformer_encoder/conv1d/bias8transformer_model_17/transformer_encoder/conv1d_1/kernel6transformer_model_17/transformer_encoder/conv1d_1/biasDtransformer_model_17/transformer_encoder/layer_normalization_1/gammaCtransformer_model_17/transformer_encoder/layer_normalization_1/beta!transformer_model_17/dense/kerneltransformer_model_17/dense/bias#transformer_model_17/dense_1/kernel!transformer_model_17/dense_1/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_295859
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
๗*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameLtransformer_model_17/positional_embedding/dense_2/kernel/Read/ReadVariableOpJtransformer_model_17/positional_embedding/dense_2/bias/Read/ReadVariableOp^transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/Read/ReadVariableOp\transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/Read/ReadVariableOp^transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/Read/ReadVariableOpctransformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/Read/ReadVariableOpatransformer_model_17/transformer_encoder/multi_head_attention/projection_bias/Read/ReadVariableOpVtransformer_model_17/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpUtransformer_model_17/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpJtransformer_model_17/transformer_encoder/conv1d/kernel/Read/ReadVariableOpHtransformer_model_17/transformer_encoder/conv1d/bias/Read/ReadVariableOpLtransformer_model_17/transformer_encoder/conv1d_1/kernel/Read/ReadVariableOpJtransformer_model_17/transformer_encoder/conv1d_1/bias/Read/ReadVariableOpXtransformer_model_17/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpWtransformer_model_17/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOp5transformer_model_17/dense/kernel/Read/ReadVariableOp3transformer_model_17/dense/bias/Read/ReadVariableOp7transformer_model_17/dense_1/kernel/Read/ReadVariableOp5transformer_model_17/dense_1/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpSAdam/transformer_model_17/positional_embedding/dense_2/kernel/m/Read/ReadVariableOpQAdam/transformer_model_17/positional_embedding/dense_2/bias/m/Read/ReadVariableOpeAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/m/Read/ReadVariableOpcAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/m/Read/ReadVariableOpeAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/m/Read/ReadVariableOpjAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/m/Read/ReadVariableOphAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/m/Read/ReadVariableOp]Adam/transformer_model_17/transformer_encoder/layer_normalization/gamma/m/Read/ReadVariableOp\Adam/transformer_model_17/transformer_encoder/layer_normalization/beta/m/Read/ReadVariableOpQAdam/transformer_model_17/transformer_encoder/conv1d/kernel/m/Read/ReadVariableOpOAdam/transformer_model_17/transformer_encoder/conv1d/bias/m/Read/ReadVariableOpSAdam/transformer_model_17/transformer_encoder/conv1d_1/kernel/m/Read/ReadVariableOpQAdam/transformer_model_17/transformer_encoder/conv1d_1/bias/m/Read/ReadVariableOp_Adam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/m/Read/ReadVariableOp^Adam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/m/Read/ReadVariableOp<Adam/transformer_model_17/dense/kernel/m/Read/ReadVariableOp:Adam/transformer_model_17/dense/bias/m/Read/ReadVariableOp>Adam/transformer_model_17/dense_1/kernel/m/Read/ReadVariableOp<Adam/transformer_model_17/dense_1/bias/m/Read/ReadVariableOpSAdam/transformer_model_17/positional_embedding/dense_2/kernel/v/Read/ReadVariableOpQAdam/transformer_model_17/positional_embedding/dense_2/bias/v/Read/ReadVariableOpeAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/v/Read/ReadVariableOpcAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/v/Read/ReadVariableOpeAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/v/Read/ReadVariableOpjAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/v/Read/ReadVariableOphAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/v/Read/ReadVariableOp]Adam/transformer_model_17/transformer_encoder/layer_normalization/gamma/v/Read/ReadVariableOp\Adam/transformer_model_17/transformer_encoder/layer_normalization/beta/v/Read/ReadVariableOpQAdam/transformer_model_17/transformer_encoder/conv1d/kernel/v/Read/ReadVariableOpOAdam/transformer_model_17/transformer_encoder/conv1d/bias/v/Read/ReadVariableOpSAdam/transformer_model_17/transformer_encoder/conv1d_1/kernel/v/Read/ReadVariableOpQAdam/transformer_model_17/transformer_encoder/conv1d_1/bias/v/Read/ReadVariableOp_Adam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/v/Read/ReadVariableOp^Adam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/v/Read/ReadVariableOp<Adam/transformer_model_17/dense/kernel/v/Read/ReadVariableOp:Adam/transformer_model_17/dense/bias/v/Read/ReadVariableOp>Adam/transformer_model_17/dense_1/kernel/v/Read/ReadVariableOp<Adam/transformer_model_17/dense_1/bias/v/Read/ReadVariableOpConst_1*Q
TinJ
H2F	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_296921
? 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename8transformer_model_17/positional_embedding/dense_2/kernel6transformer_model_17/positional_embedding/dense_2/biasJtransformer_model_17/transformer_encoder/multi_head_attention/query_kernelHtransformer_model_17/transformer_encoder/multi_head_attention/key_kernelJtransformer_model_17/transformer_encoder/multi_head_attention/value_kernelOtransformer_model_17/transformer_encoder/multi_head_attention/projection_kernelMtransformer_model_17/transformer_encoder/multi_head_attention/projection_biasBtransformer_model_17/transformer_encoder/layer_normalization/gammaAtransformer_model_17/transformer_encoder/layer_normalization/beta6transformer_model_17/transformer_encoder/conv1d/kernel4transformer_model_17/transformer_encoder/conv1d/bias8transformer_model_17/transformer_encoder/conv1d_1/kernel6transformer_model_17/transformer_encoder/conv1d_1/biasDtransformer_model_17/transformer_encoder/layer_normalization_1/gammaCtransformer_model_17/transformer_encoder/layer_normalization_1/beta!transformer_model_17/dense/kerneltransformer_model_17/dense/bias#transformer_model_17/dense_1/kernel!transformer_model_17/dense_1/biasbeta_1beta_2decaylearning_rate	Adam/itertotal_2count_2total_1count_1totalcount?Adam/transformer_model_17/positional_embedding/dense_2/kernel/m=Adam/transformer_model_17/positional_embedding/dense_2/bias/mQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/mOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/mQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/mVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/mTAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/mIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/mHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/m=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/m;Adam/transformer_model_17/transformer_encoder/conv1d/bias/m?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/m=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/mKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/mJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/m(Adam/transformer_model_17/dense/kernel/m&Adam/transformer_model_17/dense/bias/m*Adam/transformer_model_17/dense_1/kernel/m(Adam/transformer_model_17/dense_1/bias/m?Adam/transformer_model_17/positional_embedding/dense_2/kernel/v=Adam/transformer_model_17/positional_embedding/dense_2/bias/vQAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/vOAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/vQAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/vVAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/vTAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/vIAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/vHAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/v=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/v;Adam/transformer_model_17/transformer_encoder/conv1d/bias/v?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/v=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/vKAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/vJAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/v(Adam/transformer_model_17/dense/kernel/v&Adam/transformer_model_17/dense/bias/v*Adam/transformer_model_17/dense_1/kernel/v(Adam/transformer_model_17/dense_1/bias/v*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_297135ฅ

ค
$__inference_signature_wrapper_295859
input_1
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	!
	unknown_9:

unknown_10:	"

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	

unknown_18:
identityขStatefulPartitionedCallฌ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_295008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1:&"
 
_output_shapes
:


p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_295018

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ข
D
(__inference_dropout_layer_call_fn_296671

inputs
identityฒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_295230a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ฺ
a
C__inference_dropout_layer_call_and_return_conditional_losses_295230

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ฐ
ฏ
5__inference_transformer_model_17_layer_call_fn_295949
x
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	!
	unknown_9:

unknown_10:	"

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	

unknown_18:
identityขStatefulPartitionedCallี
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295618o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:

๙	
b
C__inference_dropout_layer_call_and_return_conditional_losses_296693

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ง
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
4
ฑ
P__inference_positional_embedding_layer_call_and_return_conditional_losses_295082
x<
)dense_2_tensordot_readvariableop_resource:	6
'dense_2_biasadd_readvariableop_resource:	
unknown
identityขdense_2/BiasAdd/ReadVariableOpข dense_2/Tensordot/ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       H
dense_2/Tensordot/ShapeShapex*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ฿
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ผ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposex!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????ข
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ฃ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????d
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ว
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????I
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 7
SqrtSqrtCast:y:0*
T0*
_output_shapes
: e
mulMuldense_2/BiasAdd:output:0Sqrt:y:0*
T0*,
_output_shapes
:?????????G
ConstConst*
_output_shapes
: *
dtype0*
value	B : I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :Y
strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : Y
strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 
strided_slice_1/stackPack strided_slice_1/stack/0:output:0Const:output:0 strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : [
strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ญ
strided_slice_1/stack_1Pack"strided_slice_1/stack_1/0:output:0strided_slice:output:0"strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :[
strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :ง
strided_slice_1/stack_2Pack"strided_slice_1/stack_2/0:output:0Const_1:output:0"strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:?
strided_slice_1StridedSliceunknownstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskf
addAddV2mul:z:0strided_slice_1:output:0*
T0*,
_output_shapes
:?????????[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:?????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : :
2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:

ฃ

๕
C__inference_dense_1_layer_call_and_return_conditional_losses_296385

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ฒ|

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295179

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identityขconv1d/BiasAdd/ReadVariableOpข)conv1d/Conv1D/ExpandDims_1/ReadVariableOpขconv1d_1/BiasAdd/ReadVariableOpข+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpข,layer_normalization/batchnorm/ReadVariableOpข0layer_normalization/batchnorm/mul/ReadVariableOpข.layer_normalization_1/batchnorm/ReadVariableOpข2layer_normalization_1/batchnorm/mul/ReadVariableOpข'multi_head_attention/add/ReadVariableOpข1multi_head_attention/einsum/Einsum/ReadVariableOpข3multi_head_attention/einsum_1/Einsum/ReadVariableOpข3multi_head_attention/einsum_2/Einsum/ReadVariableOpข3multi_head_attention/einsum_5/Einsum/ReadVariableOpฒ
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ิ
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...NI,HIO->...NHOถ
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHOถ
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO_
multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Cg
multi_head_attention/SqrtSqrt#multi_head_attention/Const:output:0*
T0*
_output_shapes
: ฎ
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:?????????็
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:?????????*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:?????????
'multi_head_attention/dropout_1/IdentityIdentity&multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:?????????๘
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/Identity:output:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:?????????*#
equation...HNM,...MHI->...NHIถ
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0๛
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0ธ
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????s
dropout_2/IdentityIdentitymulti_head_attention/add:z:0*
T0*,
_output_shapes
:?????????|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ษ
 layer_normalization/moments/meanMeandropout_2/Identity:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????ษ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/Identity:output:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:็
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ฝ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ง
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ย
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ฅ
#layer_normalization/batchnorm/mul_1Muldropout_2/Identity:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0พ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????t
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:?????????g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ข
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ท
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ร
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????p
dropout_3/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????ฉ
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ฆ
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ฝ
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ษ
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ห
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:?????????ห
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ํ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ร
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ซ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ศ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ง
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ฃ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ฤ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????y
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????ช
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????: : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2R
'multi_head_attention/add/ReadVariableOp'multi_head_attention/add/ReadVariableOp2f
1multi_head_attention/einsum/Einsum/ReadVariableOp1multi_head_attention/einsum/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_1/Einsum/ReadVariableOp3multi_head_attention/einsum_1/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_2/Einsum/ReadVariableOp3multi_head_attention/einsum_2/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_5/Einsum/ReadVariableOp3multi_head_attention/einsum_5/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ฦ

&__inference_dense_layer_call_fn_296655

inputs
unknown:

	unknown_0:	
identityขStatefulPartitionedCallฺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_295219p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ผ'
	
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295618
x.
positional_embedding_295571:	*
positional_embedding_295573:	
positional_embedding_2955752
transformer_encoder_295578:2
transformer_encoder_295580:2
transformer_encoder_295582:2
transformer_encoder_295584:)
transformer_encoder_295586:	)
transformer_encoder_295588:	)
transformer_encoder_295590:	2
transformer_encoder_295592:)
transformer_encoder_295594:	2
transformer_encoder_295596:)
transformer_encoder_295598:	)
transformer_encoder_295600:	)
transformer_encoder_295602:	 
dense_295606:

dense_295608:	!
dense_1_295612:	
dense_1_295614:
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdropout/StatefulPartitionedCallข,positional_embedding/StatefulPartitionedCallข+transformer_encoder/StatefulPartitionedCallม
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallxpositional_embedding_295571positional_embedding_295573positional_embedding_295575*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_295082
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_295578transformer_encoder_295580transformer_encoder_295582transformer_encoder_295584transformer_encoder_295586transformer_encoder_295588transformer_encoder_295590transformer_encoder_295592transformer_encoder_295594transformer_encoder_295596transformer_encoder_295598transformer_encoder_295600transformer_encoder_295602*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295480
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_295018
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_295606dense_295608*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_295219๊
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_295323
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_295612dense_1_295614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_295243w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:

อ๖
ช
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296105
xQ
>positional_embedding_dense_2_tensordot_readvariableop_resource:	K
<positional_embedding_dense_2_biasadd_readvariableop_resource:	
positional_embedding_295987f
Ntransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource:S
Dtransformer_encoder_multi_head_attention_add_readvariableop_resource:	\
Mtransformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource:	X
Itransformer_encoder_layer_normalization_batchnorm_readvariableop_resource:	^
Ftransformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource:I
:transformer_encoder_conv1d_biasadd_readvariableop_resource:	`
Htransformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource:K
<transformer_encoder_conv1d_1_biasadd_readvariableop_resource:	^
Otransformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource:	Z
Ktransformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identityขdense/BiasAdd/ReadVariableOpขdense/MatMul/ReadVariableOpขdense_1/BiasAdd/ReadVariableOpขdense_1/MatMul/ReadVariableOpข3positional_embedding/dense_2/BiasAdd/ReadVariableOpข5positional_embedding/dense_2/Tensordot/ReadVariableOpข1transformer_encoder/conv1d/BiasAdd/ReadVariableOpข=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpข3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpข?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpข@transformer_encoder/layer_normalization/batchnorm/ReadVariableOpขDtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpขBtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpขFtransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpข;transformer_encoder/multi_head_attention/add/ReadVariableOpขEtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpขGtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpขGtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpขGtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpK
positional_embedding/ShapeShapex*
T0*
_output_shapes
:r
(positional_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*positional_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*positional_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:บ
"positional_embedding/strided_sliceStridedSlice#positional_embedding/Shape:output:01positional_embedding/strided_slice/stack:output:03positional_embedding/strided_slice/stack_1:output:03positional_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskต
5positional_embedding/dense_2/Tensordot/ReadVariableOpReadVariableOp>positional_embedding_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0u
+positional_embedding/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+positional_embedding/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
,positional_embedding/dense_2/Tensordot/ShapeShapex*
T0*
_output_shapes
:v
4positional_embedding/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ฏ
/positional_embedding/dense_2/Tensordot/GatherV2GatherV25positional_embedding/dense_2/Tensordot/Shape:output:04positional_embedding/dense_2/Tensordot/free:output:0=positional_embedding/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6positional_embedding/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ณ
1positional_embedding/dense_2/Tensordot/GatherV2_1GatherV25positional_embedding/dense_2/Tensordot/Shape:output:04positional_embedding/dense_2/Tensordot/axes:output:0?positional_embedding/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,positional_embedding/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ล
+positional_embedding/dense_2/Tensordot/ProdProd8positional_embedding/dense_2/Tensordot/GatherV2:output:05positional_embedding/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.positional_embedding/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ห
-positional_embedding/dense_2/Tensordot/Prod_1Prod:positional_embedding/dense_2/Tensordot/GatherV2_1:output:07positional_embedding/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2positional_embedding/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-positional_embedding/dense_2/Tensordot/concatConcatV24positional_embedding/dense_2/Tensordot/free:output:04positional_embedding/dense_2/Tensordot/axes:output:0;positional_embedding/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ะ
,positional_embedding/dense_2/Tensordot/stackPack4positional_embedding/dense_2/Tensordot/Prod:output:06positional_embedding/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ฎ
0positional_embedding/dense_2/Tensordot/transpose	Transposex6positional_embedding/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????แ
.positional_embedding/dense_2/Tensordot/ReshapeReshape4positional_embedding/dense_2/Tensordot/transpose:y:05positional_embedding/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????โ
-positional_embedding/dense_2/Tensordot/MatMulMatMul7positional_embedding/dense_2/Tensordot/Reshape:output:0=positional_embedding/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????y
.positional_embedding/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:v
4positional_embedding/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/positional_embedding/dense_2/Tensordot/concat_1ConcatV28positional_embedding/dense_2/Tensordot/GatherV2:output:07positional_embedding/dense_2/Tensordot/Const_2:output:0=positional_embedding/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&positional_embedding/dense_2/TensordotReshape7positional_embedding/dense_2/Tensordot/MatMul:product:08positional_embedding/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????ญ
3positional_embedding/dense_2/BiasAdd/ReadVariableOpReadVariableOp<positional_embedding_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ิ
$positional_embedding/dense_2/BiasAddBiasAdd/positional_embedding/dense_2/Tensordot:output:0;positional_embedding/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????^
positional_embedding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :w
positional_embedding/CastCast$positional_embedding/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: a
positional_embedding/SqrtSqrtpositional_embedding/Cast:y:0*
T0*
_output_shapes
: ค
positional_embedding/mulMul-positional_embedding/dense_2/BiasAdd:output:0positional_embedding/Sqrt:y:0*
T0*,
_output_shapes
:?????????\
positional_embedding/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
positional_embedding/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,positional_embedding/strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : n
,positional_embedding/strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ๓
*positional_embedding/strided_slice_1/stackPack5positional_embedding/strided_slice_1/stack/0:output:0#positional_embedding/Const:output:05positional_embedding/strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:p
.positional_embedding/strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : p
.positional_embedding/strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : 
,positional_embedding/strided_slice_1/stack_1Pack7positional_embedding/strided_slice_1/stack_1/0:output:0+positional_embedding/strided_slice:output:07positional_embedding/strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:p
.positional_embedding/strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :p
.positional_embedding/strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :๛
,positional_embedding/strided_slice_1/stack_2Pack7positional_embedding/strided_slice_1/stack_2/0:output:0%positional_embedding/Const_1:output:07positional_embedding/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:ๆ
$positional_embedding/strided_slice_1StridedSlicepositional_embedding_2959873positional_embedding/strided_slice_1/stack:output:05positional_embedding/strided_slice_1/stack_1:output:05positional_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskฅ
positional_embedding/addAddV2positional_embedding/mul:z:0-positional_embedding/strided_slice_1:output:0*
T0*,
_output_shapes
:?????????ฺ
Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
6transformer_encoder/multi_head_attention/einsum/EinsumEinsumpositional_embedding/add:z:0Mtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...NI,HIO->...NHO?
Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO?
Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_2/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHOs
.transformer_encoder/multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  C
-transformer_encoder/multi_head_attention/SqrtSqrt7transformer_encoder/multi_head_attention/Const:output:0*
T0*
_output_shapes
: ๊
0transformer_encoder/multi_head_attention/truedivRealDiv?transformer_encoder/multi_head_attention/einsum/Einsum:output:01transformer_encoder/multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:?????????ฃ
8transformer_encoder/multi_head_attention/einsum_3/EinsumEinsum4transformer_encoder/multi_head_attention/truediv:z:0Atransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:?????????*#
equation...NHO,...MHO->...HNMธ
0transformer_encoder/multi_head_attention/SoftmaxSoftmaxAtransformer_encoder/multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:?????????ฝ
;transformer_encoder/multi_head_attention/dropout_1/IdentityIdentity:transformer_encoder/multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:?????????ด
8transformer_encoder/multi_head_attention/einsum_4/EinsumEinsumDtransformer_encoder/multi_head_attention/dropout_1/Identity:output:0Atransformer_encoder/multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:?????????*#
equation...HNM,...MHI->...NHI?
Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ท
8transformer_encoder/multi_head_attention/einsum_5/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_4/Einsum:output:0Otransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????*
equation...NHI,HIO->...NOฝ
;transformer_encoder/multi_head_attention/add/ReadVariableOpReadVariableOpDtransformer_encoder_multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0๔
,transformer_encoder/multi_head_attention/addAddV2Atransformer_encoder/multi_head_attention/einsum_5/Einsum:output:0Ctransformer_encoder/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????
&transformer_encoder/dropout_2/IdentityIdentity0transformer_encoder/multi_head_attention/add:z:0*
T0*,
_output_shapes
:?????????
Ftransformer_encoder/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
4transformer_encoder/layer_normalization/moments/meanMean/transformer_encoder/dropout_2/Identity:output:0Otransformer_encoder/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(ม
<transformer_encoder/layer_normalization/moments/StopGradientStopGradient=transformer_encoder/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????
Atransformer_encoder/layer_normalization/moments/SquaredDifferenceSquaredDifference/transformer_encoder/dropout_2/Identity:output:0Etransformer_encoder/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
Jtransformer_encoder/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ฃ
8transformer_encoder/layer_normalization/moments/varianceMeanEtransformer_encoder/layer_normalization/moments/SquaredDifference:z:0Stransformer_encoder/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(|
7transformer_encoder/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75๙
5transformer_encoder/layer_normalization/batchnorm/addAddV2Atransformer_encoder/layer_normalization/moments/variance:output:0@transformer_encoder/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????ฑ
7transformer_encoder/layer_normalization/batchnorm/RsqrtRsqrt9transformer_encoder/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ฯ
Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0?
5transformer_encoder/layer_normalization/batchnorm/mulMul;transformer_encoder/layer_normalization/batchnorm/Rsqrt:y:0Ltransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????แ
7transformer_encoder/layer_normalization/batchnorm/mul_1Mul/transformer_encoder/dropout_2/Identity:output:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????๏
7transformer_encoder/layer_normalization/batchnorm/mul_2Mul=transformer_encoder/layer_normalization/moments/mean:output:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ว
@transformer_encoder/layer_normalization/batchnorm/ReadVariableOpReadVariableOpItransformer_encoder_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0๚
5transformer_encoder/layer_normalization/batchnorm/subSubHtransformer_encoder/layer_normalization/batchnorm/ReadVariableOp:value:0;transformer_encoder/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????๏
7transformer_encoder/layer_normalization/batchnorm/add_1AddV2;transformer_encoder/layer_normalization/batchnorm/mul_1:z:09transformer_encoder/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ฒ
transformer_encoder/addAddV2;transformer_encoder/layer_normalization/batchnorm/add_1:z:0positional_embedding/add:z:0*
T0*,
_output_shapes
:?????????{
0transformer_encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????อ
,transformer_encoder/conv1d/Conv1D/ExpandDims
ExpandDimstransformer_encoder/add:z:09transformer_encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ส
=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpFtransformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0t
2transformer_encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ๓
.transformer_encoder/conv1d/Conv1D/ExpandDims_1
ExpandDimsEtransformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0;transformer_encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:?
!transformer_encoder/conv1d/Conv1DConv2D5transformer_encoder/conv1d/Conv1D/ExpandDims:output:07transformer_encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
ท
)transformer_encoder/conv1d/Conv1D/SqueezeSqueeze*transformer_encoder/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????ฉ
1transformer_encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp:transformer_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ำ
"transformer_encoder/conv1d/BiasAddBiasAdd2transformer_encoder/conv1d/Conv1D/Squeeze:output:09transformer_encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????
transformer_encoder/conv1d/ReluRelu+transformer_encoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????
&transformer_encoder/dropout_3/IdentityIdentity-transformer_encoder/conv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????}
2transformer_encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????ๅ
.transformer_encoder/conv1d_1/Conv1D/ExpandDims
ExpandDims/transformer_encoder/dropout_3/Identity:output:0;transformer_encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ฮ
?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpHtransformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0v
4transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ๙
0transformer_encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDimsGtransformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0=transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:
#transformer_encoder/conv1d_1/Conv1DConv2D7transformer_encoder/conv1d_1/Conv1D/ExpandDims:output:09transformer_encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
ป
+transformer_encoder/conv1d_1/Conv1D/SqueezeSqueeze,transformer_encoder/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????ญ
3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ู
$transformer_encoder/conv1d_1/BiasAddBiasAdd4transformer_encoder/conv1d_1/Conv1D/Squeeze:output:0;transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????
Htransformer_encoder/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
6transformer_encoder/layer_normalization_1/moments/meanMean-transformer_encoder/conv1d_1/BiasAdd:output:0Qtransformer_encoder/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(ล
>transformer_encoder/layer_normalization_1/moments/StopGradientStopGradient?transformer_encoder/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:?????????
Ctransformer_encoder/layer_normalization_1/moments/SquaredDifferenceSquaredDifference-transformer_encoder/conv1d_1/BiasAdd:output:0Gtransformer_encoder/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
Ltransformer_encoder/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ฉ
:transformer_encoder/layer_normalization_1/moments/varianceMeanGtransformer_encoder/layer_normalization_1/moments/SquaredDifference:z:0Utransformer_encoder/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(~
9transformer_encoder/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75?
7transformer_encoder/layer_normalization_1/batchnorm/addAddV2Ctransformer_encoder/layer_normalization_1/moments/variance:output:0Btransformer_encoder/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????ต
9transformer_encoder/layer_normalization_1/batchnorm/RsqrtRsqrt;transformer_encoder/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ำ
Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/mulMul=transformer_encoder/layer_normalization_1/batchnorm/Rsqrt:y:0Ntransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ใ
9transformer_encoder/layer_normalization_1/batchnorm/mul_1Mul-transformer_encoder/conv1d_1/BiasAdd:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????๕
9transformer_encoder/layer_normalization_1/batchnorm/mul_2Mul?transformer_encoder/layer_normalization_1/moments/mean:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ห
Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpKtransformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/subSubJtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp:value:0=transformer_encoder/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????๕
9transformer_encoder/layer_normalization_1/batchnorm/add_1AddV2=transformer_encoder/layer_normalization_1/batchnorm/mul_1:z:0;transformer_encoder/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ต
transformer_encoder/add_1AddV2transformer_encoder/add:z:0=transformer_encoder/layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :ฑ
global_average_pooling1d/MeanMeantransformer_encoder/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????i
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp4^positional_embedding/dense_2/BiasAdd/ReadVariableOp6^positional_embedding/dense_2/Tensordot/ReadVariableOp2^transformer_encoder/conv1d/BiasAdd/ReadVariableOp>^transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4^transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp@^transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpA^transformer_encoder/layer_normalization/batchnorm/ReadVariableOpE^transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpC^transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpG^transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp<^transformer_encoder/multi_head_attention/add/ReadVariableOpF^transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2j
3positional_embedding/dense_2/BiasAdd/ReadVariableOp3positional_embedding/dense_2/BiasAdd/ReadVariableOp2n
5positional_embedding/dense_2/Tensordot/ReadVariableOp5positional_embedding/dense_2/Tensordot/ReadVariableOp2f
1transformer_encoder/conv1d/BiasAdd/ReadVariableOp1transformer_encoder/conv1d/BiasAdd/ReadVariableOp2~
=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2j
3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp2
?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2
@transformer_encoder/layer_normalization/batchnorm/ReadVariableOp@transformer_encoder/layer_normalization/batchnorm/ReadVariableOp2
Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpDtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp2
Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpBtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp2
Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpFtransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp2z
;transformer_encoder/multi_head_attention/add/ReadVariableOp;transformer_encoder/multi_head_attention/add/ReadVariableOp2
Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpEtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp2
Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpGtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp2
Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpGtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp2
Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpGtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:

4
ฑ
P__inference_positional_embedding_layer_call_and_return_conditional_losses_296354
x<
)dense_2_tensordot_readvariableop_resource:	6
'dense_2_biasadd_readvariableop_resource:	
unknown
identityขdense_2/BiasAdd/ReadVariableOpข dense_2/Tensordot/ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ั
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       H
dense_2/Tensordot/ShapeShapex*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ฿
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ผ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposex!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????ข
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ฃ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????d
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ว
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????I
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 7
SqrtSqrtCast:y:0*
T0*
_output_shapes
: e
mulMuldense_2/BiasAdd:output:0Sqrt:y:0*
T0*,
_output_shapes
:?????????G
ConstConst*
_output_shapes
: *
dtype0*
value	B : I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :Y
strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : Y
strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 
strided_slice_1/stackPack strided_slice_1/stack/0:output:0Const:output:0 strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : [
strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ญ
strided_slice_1/stack_1Pack"strided_slice_1/stack_1/0:output:0strided_slice:output:0"strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:[
strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :[
strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :ง
strided_slice_1/stack_2Pack"strided_slice_1/stack_2/0:output:0Const_1:output:0"strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:?
strided_slice_1StridedSliceunknownstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskf
addAddV2mul:z:0strided_slice_1:output:0*
T0*,
_output_shapes
:?????????[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:?????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : :
2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:

๊
๕
4__inference_transformer_encoder_layer_call_fn_296447

inputs
unknown:!
	unknown_0:!
	unknown_1:!
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	!
	unknown_6:
	unknown_7:	!
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295480t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ค

๕
A__inference_dense_layer_call_and_return_conditional_losses_295219

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๔
a
(__inference_dropout_layer_call_fn_296676

inputs
identityขStatefulPartitionedCallย
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_295323p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ย
ต
5__inference_transformer_model_17_layer_call_fn_295706
input_1
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	!
	unknown_9:

unknown_10:	"

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	

unknown_18:
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295618o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1:&"
 
_output_shapes
:

ดท
ด
!__inference__wrapped_model_295008
input_1f
Stransformer_model_17_positional_embedding_dense_2_tensordot_readvariableop_resource:	`
Qtransformer_model_17_positional_embedding_dense_2_biasadd_readvariableop_resource:	4
0transformer_model_17_positional_embedding_294890{
ctransformer_model_17_transformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource:}
etransformer_model_17_transformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource:}
etransformer_model_17_transformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource:}
etransformer_model_17_transformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource:h
Ytransformer_model_17_transformer_encoder_multi_head_attention_add_readvariableop_resource:	q
btransformer_model_17_transformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource:	m
^transformer_model_17_transformer_encoder_layer_normalization_batchnorm_readvariableop_resource:	s
[transformer_model_17_transformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource:^
Otransformer_model_17_transformer_encoder_conv1d_biasadd_readvariableop_resource:	u
]transformer_model_17_transformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource:`
Qtransformer_model_17_transformer_encoder_conv1d_1_biasadd_readvariableop_resource:	s
dtransformer_model_17_transformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource:	o
`transformer_model_17_transformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource:	M
9transformer_model_17_dense_matmul_readvariableop_resource:
I
:transformer_model_17_dense_biasadd_readvariableop_resource:	N
;transformer_model_17_dense_1_matmul_readvariableop_resource:	J
<transformer_model_17_dense_1_biasadd_readvariableop_resource:
identityข1transformer_model_17/dense/BiasAdd/ReadVariableOpข0transformer_model_17/dense/MatMul/ReadVariableOpข3transformer_model_17/dense_1/BiasAdd/ReadVariableOpข2transformer_model_17/dense_1/MatMul/ReadVariableOpขHtransformer_model_17/positional_embedding/dense_2/BiasAdd/ReadVariableOpขJtransformer_model_17/positional_embedding/dense_2/Tensordot/ReadVariableOpขFtransformer_model_17/transformer_encoder/conv1d/BiasAdd/ReadVariableOpขRtransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpขHtransformer_model_17/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpขTtransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpขUtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/ReadVariableOpขYtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpขWtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpข[transformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpขPtransformer_model_17/transformer_encoder/multi_head_attention/add/ReadVariableOpขZtransformer_model_17/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpข\transformer_model_17/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpข\transformer_model_17/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpข\transformer_model_17/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpf
/transformer_model_17/positional_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:
=transformer_model_17/positional_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?transformer_model_17/positional_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?transformer_model_17/positional_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ฃ
7transformer_model_17/positional_embedding/strided_sliceStridedSlice8transformer_model_17/positional_embedding/Shape:output:0Ftransformer_model_17/positional_embedding/strided_slice/stack:output:0Htransformer_model_17/positional_embedding/strided_slice/stack_1:output:0Htransformer_model_17/positional_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask฿
Jtransformer_model_17/positional_embedding/dense_2/Tensordot/ReadVariableOpReadVariableOpStransformer_model_17_positional_embedding_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0
@transformer_model_17/positional_embedding/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
@transformer_model_17/positional_embedding/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
Atransformer_model_17/positional_embedding/dense_2/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:
Itransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Dtransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2GatherV2Jtransformer_model_17/positional_embedding/dense_2/Tensordot/Shape:output:0Itransformer_model_17/positional_embedding/dense_2/Tensordot/free:output:0Rtransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Ktransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ftransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2_1GatherV2Jtransformer_model_17/positional_embedding/dense_2/Tensordot/Shape:output:0Itransformer_model_17/positional_embedding/dense_2/Tensordot/axes:output:0Ttransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Atransformer_model_17/positional_embedding/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
@transformer_model_17/positional_embedding/dense_2/Tensordot/ProdProdMtransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2:output:0Jtransformer_model_17/positional_embedding/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 
Ctransformer_model_17/positional_embedding/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Btransformer_model_17/positional_embedding/dense_2/Tensordot/Prod_1ProdOtransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2_1:output:0Ltransformer_model_17/positional_embedding/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Gtransformer_model_17/positional_embedding/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ไ
Btransformer_model_17/positional_embedding/dense_2/Tensordot/concatConcatV2Itransformer_model_17/positional_embedding/dense_2/Tensordot/free:output:0Itransformer_model_17/positional_embedding/dense_2/Tensordot/axes:output:0Ptransformer_model_17/positional_embedding/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
Atransformer_model_17/positional_embedding/dense_2/Tensordot/stackPackItransformer_model_17/positional_embedding/dense_2/Tensordot/Prod:output:0Ktransformer_model_17/positional_embedding/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
Etransformer_model_17/positional_embedding/dense_2/Tensordot/transpose	Transposeinput_1Ktransformer_model_17/positional_embedding/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:??????????
Ctransformer_model_17/positional_embedding/dense_2/Tensordot/ReshapeReshapeItransformer_model_17/positional_embedding/dense_2/Tensordot/transpose:y:0Jtransformer_model_17/positional_embedding/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????ก
Btransformer_model_17/positional_embedding/dense_2/Tensordot/MatMulMatMulLtransformer_model_17/positional_embedding/dense_2/Tensordot/Reshape:output:0Rtransformer_model_17/positional_embedding/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
Ctransformer_model_17/positional_embedding/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
Itransformer_model_17/positional_embedding/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ๏
Dtransformer_model_17/positional_embedding/dense_2/Tensordot/concat_1ConcatV2Mtransformer_model_17/positional_embedding/dense_2/Tensordot/GatherV2:output:0Ltransformer_model_17/positional_embedding/dense_2/Tensordot/Const_2:output:0Rtransformer_model_17/positional_embedding/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
;transformer_model_17/positional_embedding/dense_2/TensordotReshapeLtransformer_model_17/positional_embedding/dense_2/Tensordot/MatMul:product:0Mtransformer_model_17/positional_embedding/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????ื
Htransformer_model_17/positional_embedding/dense_2/BiasAdd/ReadVariableOpReadVariableOpQtransformer_model_17_positional_embedding_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
9transformer_model_17/positional_embedding/dense_2/BiasAddBiasAddDtransformer_model_17/positional_embedding/dense_2/Tensordot:output:0Ptransformer_model_17/positional_embedding/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????s
0transformer_model_17/positional_embedding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :ก
.transformer_model_17/positional_embedding/CastCast9transformer_model_17/positional_embedding/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
.transformer_model_17/positional_embedding/SqrtSqrt2transformer_model_17/positional_embedding/Cast:y:0*
T0*
_output_shapes
: ใ
-transformer_model_17/positional_embedding/mulMulBtransformer_model_17/positional_embedding/dense_2/BiasAdd:output:02transformer_model_17/positional_embedding/Sqrt:y:0*
T0*,
_output_shapes
:?????????q
/transformer_model_17/positional_embedding/ConstConst*
_output_shapes
: *
dtype0*
value	B : s
1transformer_model_17/positional_embedding/Const_1Const*
_output_shapes
: *
dtype0*
value	B :
Atransformer_model_17/positional_embedding/strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 
Atransformer_model_17/positional_embedding/strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ว
?transformer_model_17/positional_embedding/strided_slice_1/stackPackJtransformer_model_17/positional_embedding/strided_slice_1/stack/0:output:08transformer_model_17/positional_embedding/Const:output:0Jtransformer_model_17/positional_embedding/strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:
Ctransformer_model_17/positional_embedding/strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 
Ctransformer_model_17/positional_embedding/strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : ี
Atransformer_model_17/positional_embedding/strided_slice_1/stack_1PackLtransformer_model_17/positional_embedding/strided_slice_1/stack_1/0:output:0@transformer_model_17/positional_embedding/strided_slice:output:0Ltransformer_model_17/positional_embedding/strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:
Ctransformer_model_17/positional_embedding/strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :
Ctransformer_model_17/positional_embedding/strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :ฯ
Atransformer_model_17/positional_embedding/strided_slice_1/stack_2PackLtransformer_model_17/positional_embedding/strided_slice_1/stack_2/0:output:0:transformer_model_17/positional_embedding/Const_1:output:0Ltransformer_model_17/positional_embedding/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:ฯ
9transformer_model_17/positional_embedding/strided_slice_1StridedSlice0transformer_model_17_positional_embedding_294890Htransformer_model_17/positional_embedding/strided_slice_1/stack:output:0Jtransformer_model_17/positional_embedding/strided_slice_1/stack_1:output:0Jtransformer_model_17/positional_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskไ
-transformer_model_17/positional_embedding/addAddV21transformer_model_17/positional_embedding/mul:z:0Btransformer_model_17/positional_embedding/strided_slice_1:output:0*
T0*,
_output_shapes
:?????????
Ztransformer_model_17/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOpctransformer_model_17_transformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ั
Ktransformer_model_17/transformer_encoder/multi_head_attention/einsum/EinsumEinsum1transformer_model_17/positional_embedding/add:z:0btransformer_model_17/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...NI,HIO->...NHO
\transformer_model_17/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOpetransformer_model_17_transformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ี
Mtransformer_model_17/transformer_encoder/multi_head_attention/einsum_1/EinsumEinsum1transformer_model_17/positional_embedding/add:z:0dtransformer_model_17/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO
\transformer_model_17/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOpetransformer_model_17_transformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ี
Mtransformer_model_17/transformer_encoder/multi_head_attention/einsum_2/EinsumEinsum1transformer_model_17/positional_embedding/add:z:0dtransformer_model_17/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO
Ctransformer_model_17/transformer_encoder/multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Cน
Btransformer_model_17/transformer_encoder/multi_head_attention/SqrtSqrtLtransformer_model_17/transformer_encoder/multi_head_attention/Const:output:0*
T0*
_output_shapes
: ฉ
Etransformer_model_17/transformer_encoder/multi_head_attention/truedivRealDivTtransformer_model_17/transformer_encoder/multi_head_attention/einsum/Einsum:output:0Ftransformer_model_17/transformer_encoder/multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:?????????โ
Mtransformer_model_17/transformer_encoder/multi_head_attention/einsum_3/EinsumEinsumItransformer_model_17/transformer_encoder/multi_head_attention/truediv:z:0Vtransformer_model_17/transformer_encoder/multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:?????????*#
equation...NHO,...MHO->...HNMโ
Etransformer_model_17/transformer_encoder/multi_head_attention/SoftmaxSoftmaxVtransformer_model_17/transformer_encoder/multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:?????????็
Ptransformer_model_17/transformer_encoder/multi_head_attention/dropout_1/IdentityIdentityOtransformer_model_17/transformer_encoder/multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:?????????๓
Mtransformer_model_17/transformer_encoder/multi_head_attention/einsum_4/EinsumEinsumYtransformer_model_17/transformer_encoder/multi_head_attention/dropout_1/Identity:output:0Vtransformer_model_17/transformer_encoder/multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:?????????*#
equation...HNM,...MHI->...NHI
\transformer_model_17/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOpetransformer_model_17_transformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0๖
Mtransformer_model_17/transformer_encoder/multi_head_attention/einsum_5/EinsumEinsumVtransformer_model_17/transformer_encoder/multi_head_attention/einsum_4/Einsum:output:0dtransformer_model_17/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????*
equation...NHI,HIO->...NO็
Ptransformer_model_17/transformer_encoder/multi_head_attention/add/ReadVariableOpReadVariableOpYtransformer_model_17_transformer_encoder_multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0ณ
Atransformer_model_17/transformer_encoder/multi_head_attention/addAddV2Vtransformer_model_17/transformer_encoder/multi_head_attention/einsum_5/Einsum:output:0Xtransformer_model_17/transformer_encoder/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ล
;transformer_model_17/transformer_encoder/dropout_2/IdentityIdentityEtransformer_model_17/transformer_encoder/multi_head_attention/add:z:0*
T0*,
_output_shapes
:?????????ฅ
[transformer_model_17/transformer_encoder/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ฤ
Itransformer_model_17/transformer_encoder/layer_normalization/moments/meanMeanDtransformer_model_17/transformer_encoder/dropout_2/Identity:output:0dtransformer_model_17/transformer_encoder/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(๋
Qtransformer_model_17/transformer_encoder/layer_normalization/moments/StopGradientStopGradientRtransformer_model_17/transformer_encoder/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????ฤ
Vtransformer_model_17/transformer_encoder/layer_normalization/moments/SquaredDifferenceSquaredDifferenceDtransformer_model_17/transformer_encoder/dropout_2/Identity:output:0Ztransformer_model_17/transformer_encoder/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????ฉ
_transformer_model_17/transformer_encoder/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:โ
Mtransformer_model_17/transformer_encoder/layer_normalization/moments/varianceMeanZtransformer_model_17/transformer_encoder/layer_normalization/moments/SquaredDifference:z:0htransformer_model_17/transformer_encoder/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
Ltransformer_model_17/transformer_encoder/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ธ
Jtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/addAddV2Vtransformer_model_17/transformer_encoder/layer_normalization/moments/variance:output:0Utransformer_model_17/transformer_encoder/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:??????????
Ltransformer_model_17/transformer_encoder/layer_normalization/batchnorm/RsqrtRsqrtNtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????๙
Ytransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpbtransformer_model_17_transformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ฝ
Jtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mulMulPtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/Rsqrt:y:0atransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
Ltransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul_1MulDtransformer_model_17/transformer_encoder/dropout_2/Identity:output:0Ntransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ฎ
Ltransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul_2MulRtransformer_model_17/transformer_encoder/layer_normalization/moments/mean:output:0Ntransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????๑
Utransformer_model_17/transformer_encoder/layer_normalization/batchnorm/ReadVariableOpReadVariableOp^transformer_model_17_transformer_encoder_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0น
Jtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/subSub]transformer_model_17/transformer_encoder/layer_normalization/batchnorm/ReadVariableOp:value:0Ptransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ฎ
Ltransformer_model_17/transformer_encoder/layer_normalization/batchnorm/add_1AddV2Ptransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul_1:z:0Ntransformer_model_17/transformer_encoder/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????๑
,transformer_model_17/transformer_encoder/addAddV2Ptransformer_model_17/transformer_encoder/layer_normalization/batchnorm/add_1:z:01transformer_model_17/positional_embedding/add:z:0*
T0*,
_output_shapes
:?????????
Etransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
Atransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims
ExpandDims0transformer_model_17/transformer_encoder/add:z:0Ntransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????๔
Rtransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp[transformer_model_17_transformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0
Gtransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ฒ
Ctransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1
ExpandDimsZtransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ptransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:พ
6transformer_model_17/transformer_encoder/conv1d/Conv1DConv2DJtransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims:output:0Ltransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
แ
>transformer_model_17/transformer_encoder/conv1d/Conv1D/SqueezeSqueeze?transformer_model_17/transformer_encoder/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????ำ
Ftransformer_model_17/transformer_encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOpOtransformer_model_17_transformer_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_model_17/transformer_encoder/conv1d/BiasAddBiasAddGtransformer_model_17/transformer_encoder/conv1d/Conv1D/Squeeze:output:0Ntransformer_model_17/transformer_encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ต
4transformer_model_17/transformer_encoder/conv1d/ReluRelu@transformer_model_17/transformer_encoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????ย
;transformer_model_17/transformer_encoder/dropout_3/IdentityIdentityBtransformer_model_17/transformer_encoder/conv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????
Gtransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????ค
Ctransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims
ExpandDimsDtransformer_model_17/transformer_encoder/dropout_3/Identity:output:0Ptransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????๘
Ttransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp]transformer_model_17_transformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0
Itransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ธ
Etransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDims\transformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Rtransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ฤ
8transformer_model_17/transformer_encoder/conv1d_1/Conv1DConv2DLtransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims:output:0Ntransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
ๅ
@transformer_model_17/transformer_encoder/conv1d_1/Conv1D/SqueezeSqueezeAtransformer_model_17/transformer_encoder/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????ื
Htransformer_model_17/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOpQtransformer_model_17_transformer_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
9transformer_model_17/transformer_encoder/conv1d_1/BiasAddBiasAddItransformer_model_17/transformer_encoder/conv1d_1/Conv1D/Squeeze:output:0Ptransformer_model_17/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ง
]transformer_model_17/transformer_encoder/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ฦ
Ktransformer_model_17/transformer_encoder/layer_normalization_1/moments/meanMeanBtransformer_model_17/transformer_encoder/conv1d_1/BiasAdd:output:0ftransformer_model_17/transformer_encoder/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(๏
Stransformer_model_17/transformer_encoder/layer_normalization_1/moments/StopGradientStopGradientTtransformer_model_17/transformer_encoder/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:?????????ฦ
Xtransformer_model_17/transformer_encoder/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceBtransformer_model_17/transformer_encoder/conv1d_1/BiasAdd:output:0\transformer_model_17/transformer_encoder/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????ซ
atransformer_model_17/transformer_encoder/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:่
Otransformer_model_17/transformer_encoder/layer_normalization_1/moments/varianceMean\transformer_model_17/transformer_encoder/layer_normalization_1/moments/SquaredDifference:z:0jtransformer_model_17/transformer_encoder/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
Ntransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75พ
Ltransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/addAddV2Xtransformer_model_17/transformer_encoder/layer_normalization_1/moments/variance:output:0Wtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????฿
Ntransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/RsqrtRsqrtPtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:??????????
[transformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpdtransformer_model_17_transformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ร
Ltransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mulMulRtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/Rsqrt:y:0ctransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ข
Ntransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul_1MulBtransformer_model_17/transformer_encoder/conv1d_1/BiasAdd:output:0Ptransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ด
Ntransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul_2MulTtransformer_model_17/transformer_encoder/layer_normalization_1/moments/mean:output:0Ptransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????๕
Wtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp`transformer_model_17_transformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ฟ
Ltransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/subSub_transformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp:value:0Rtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ด
Ntransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/add_1AddV2Rtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul_1:z:0Ptransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????๔
.transformer_model_17/transformer_encoder/add_1AddV20transformer_model_17/transformer_encoder/add:z:0Rtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????
Dtransformer_model_17/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :๐
2transformer_model_17/global_average_pooling1d/MeanMean2transformer_model_17/transformer_encoder/add_1:z:0Mtransformer_model_17/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????ฌ
0transformer_model_17/dense/MatMul/ReadVariableOpReadVariableOp9transformer_model_17_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0ี
!transformer_model_17/dense/MatMulMatMul;transformer_model_17/global_average_pooling1d/Mean:output:08transformer_model_17/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????ฉ
1transformer_model_17/dense/BiasAdd/ReadVariableOpReadVariableOp:transformer_model_17_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ศ
"transformer_model_17/dense/BiasAddBiasAdd+transformer_model_17/dense/MatMul:product:09transformer_model_17/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
transformer_model_17/dense/ReluRelu+transformer_model_17/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
%transformer_model_17/dropout/IdentityIdentity-transformer_model_17/dense/Relu:activations:0*
T0*(
_output_shapes
:?????????ฏ
2transformer_model_17/dense_1/MatMul/ReadVariableOpReadVariableOp;transformer_model_17_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0ห
#transformer_model_17/dense_1/MatMulMatMul.transformer_model_17/dropout/Identity:output:0:transformer_model_17/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????ฌ
3transformer_model_17/dense_1/BiasAdd/ReadVariableOpReadVariableOp<transformer_model_17_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0อ
$transformer_model_17/dense_1/BiasAddBiasAdd-transformer_model_17/dense_1/MatMul:product:0;transformer_model_17/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
$transformer_model_17/dense_1/SoftmaxSoftmax-transformer_model_17/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
IdentityIdentity.transformer_model_17/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????ฉ
NoOpNoOp2^transformer_model_17/dense/BiasAdd/ReadVariableOp1^transformer_model_17/dense/MatMul/ReadVariableOp4^transformer_model_17/dense_1/BiasAdd/ReadVariableOp3^transformer_model_17/dense_1/MatMul/ReadVariableOpI^transformer_model_17/positional_embedding/dense_2/BiasAdd/ReadVariableOpK^transformer_model_17/positional_embedding/dense_2/Tensordot/ReadVariableOpG^transformer_model_17/transformer_encoder/conv1d/BiasAdd/ReadVariableOpS^transformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpI^transformer_model_17/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpU^transformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpV^transformer_model_17/transformer_encoder/layer_normalization/batchnorm/ReadVariableOpZ^transformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpX^transformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp\^transformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpQ^transformer_model_17/transformer_encoder/multi_head_attention/add/ReadVariableOp[^transformer_model_17/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp]^transformer_model_17/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp]^transformer_model_17/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp]^transformer_model_17/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 2f
1transformer_model_17/dense/BiasAdd/ReadVariableOp1transformer_model_17/dense/BiasAdd/ReadVariableOp2d
0transformer_model_17/dense/MatMul/ReadVariableOp0transformer_model_17/dense/MatMul/ReadVariableOp2j
3transformer_model_17/dense_1/BiasAdd/ReadVariableOp3transformer_model_17/dense_1/BiasAdd/ReadVariableOp2h
2transformer_model_17/dense_1/MatMul/ReadVariableOp2transformer_model_17/dense_1/MatMul/ReadVariableOp2
Htransformer_model_17/positional_embedding/dense_2/BiasAdd/ReadVariableOpHtransformer_model_17/positional_embedding/dense_2/BiasAdd/ReadVariableOp2
Jtransformer_model_17/positional_embedding/dense_2/Tensordot/ReadVariableOpJtransformer_model_17/positional_embedding/dense_2/Tensordot/ReadVariableOp2
Ftransformer_model_17/transformer_encoder/conv1d/BiasAdd/ReadVariableOpFtransformer_model_17/transformer_encoder/conv1d/BiasAdd/ReadVariableOp2จ
Rtransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpRtransformer_model_17/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2
Htransformer_model_17/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpHtransformer_model_17/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp2ฌ
Ttransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpTtransformer_model_17/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2ฎ
Utransformer_model_17/transformer_encoder/layer_normalization/batchnorm/ReadVariableOpUtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/ReadVariableOp2ถ
Ytransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpYtransformer_model_17/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp2ฒ
Wtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpWtransformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp2บ
[transformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp[transformer_model_17/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp2ค
Ptransformer_model_17/transformer_encoder/multi_head_attention/add/ReadVariableOpPtransformer_model_17/transformer_encoder/multi_head_attention/add/ReadVariableOp2ธ
Ztransformer_model_17/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpZtransformer_model_17/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp2ผ
\transformer_model_17/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp\transformer_model_17/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp2ผ
\transformer_model_17/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp\transformer_model_17/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp2ผ
\transformer_model_17/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp\transformer_model_17/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1:&"
 
_output_shapes
:

ย
ต
5__inference_transformer_model_17_layer_call_fn_295293
input_1
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	!
	unknown_9:

unknown_10:	"

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	

unknown_18:
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295250o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1:&"
 
_output_shapes
:

ฺ
a
C__inference_dropout_layer_call_and_return_conditional_losses_296681

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๙	
b
C__inference_dropout_layer_call_and_return_conditional_losses_295323

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ง
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๑ฃ
ํ.
__inference__traced_save_296921
file_prefixW
Ssavev2_transformer_model_17_positional_embedding_dense_2_kernel_read_readvariableopU
Qsavev2_transformer_model_17_positional_embedding_dense_2_bias_read_readvariableopi
esavev2_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopg
csavev2_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopi
esavev2_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopn
jsavev2_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_read_readvariableopl
hsavev2_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_read_readvariableopa
]savev2_transformer_model_17_transformer_encoder_layer_normalization_gamma_read_readvariableop`
\savev2_transformer_model_17_transformer_encoder_layer_normalization_beta_read_readvariableopU
Qsavev2_transformer_model_17_transformer_encoder_conv1d_kernel_read_readvariableopS
Osavev2_transformer_model_17_transformer_encoder_conv1d_bias_read_readvariableopW
Ssavev2_transformer_model_17_transformer_encoder_conv1d_1_kernel_read_readvariableopU
Qsavev2_transformer_model_17_transformer_encoder_conv1d_1_bias_read_readvariableopc
_savev2_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_read_readvariableopb
^savev2_transformer_model_17_transformer_encoder_layer_normalization_1_beta_read_readvariableop@
<savev2_transformer_model_17_dense_kernel_read_readvariableop>
:savev2_transformer_model_17_dense_bias_read_readvariableopB
>savev2_transformer_model_17_dense_1_kernel_read_readvariableop@
<savev2_transformer_model_17_dense_1_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop^
Zsavev2_adam_transformer_model_17_positional_embedding_dense_2_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_model_17_positional_embedding_dense_2_bias_m_read_readvariableopp
lsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_m_read_readvariableopn
jsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_m_read_readvariableopp
lsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_m_read_readvariableopu
qsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_m_read_readvariableops
osavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_m_read_readvariableoph
dsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_m_read_readvariableopg
csavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_m_read_readvariableop\
Xsavev2_adam_transformer_model_17_transformer_encoder_conv1d_kernel_m_read_readvariableopZ
Vsavev2_adam_transformer_model_17_transformer_encoder_conv1d_bias_m_read_readvariableop^
Zsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_m_read_readvariableopj
fsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_m_read_readvariableopi
esavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_m_read_readvariableopG
Csavev2_adam_transformer_model_17_dense_kernel_m_read_readvariableopE
Asavev2_adam_transformer_model_17_dense_bias_m_read_readvariableopI
Esavev2_adam_transformer_model_17_dense_1_kernel_m_read_readvariableopG
Csavev2_adam_transformer_model_17_dense_1_bias_m_read_readvariableop^
Zsavev2_adam_transformer_model_17_positional_embedding_dense_2_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_model_17_positional_embedding_dense_2_bias_v_read_readvariableopp
lsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_v_read_readvariableopn
jsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_v_read_readvariableopp
lsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_v_read_readvariableopu
qsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_v_read_readvariableops
osavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_v_read_readvariableoph
dsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_v_read_readvariableopg
csavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_v_read_readvariableop\
Xsavev2_adam_transformer_model_17_transformer_encoder_conv1d_kernel_v_read_readvariableopZ
Vsavev2_adam_transformer_model_17_transformer_encoder_conv1d_bias_v_read_readvariableop^
Zsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_v_read_readvariableopj
fsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_v_read_readvariableopi
esavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_v_read_readvariableopG
Csavev2_adam_transformer_model_17_dense_kernel_v_read_readvariableopE
Asavev2_adam_transformer_model_17_dense_bias_v_read_readvariableopI
Esavev2_adam_transformer_model_17_dense_1_kernel_v_read_readvariableopG
Csavev2_adam_transformer_model_17_dense_1_bias_v_read_readvariableop
savev2_const_1

identity_1ขMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
value?B๚EB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH๚
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
valueBEB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ๅ-
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Ssavev2_transformer_model_17_positional_embedding_dense_2_kernel_read_readvariableopQsavev2_transformer_model_17_positional_embedding_dense_2_bias_read_readvariableopesavev2_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopcsavev2_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopesavev2_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopjsavev2_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_read_readvariableophsavev2_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_read_readvariableop]savev2_transformer_model_17_transformer_encoder_layer_normalization_gamma_read_readvariableop\savev2_transformer_model_17_transformer_encoder_layer_normalization_beta_read_readvariableopQsavev2_transformer_model_17_transformer_encoder_conv1d_kernel_read_readvariableopOsavev2_transformer_model_17_transformer_encoder_conv1d_bias_read_readvariableopSsavev2_transformer_model_17_transformer_encoder_conv1d_1_kernel_read_readvariableopQsavev2_transformer_model_17_transformer_encoder_conv1d_1_bias_read_readvariableop_savev2_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_read_readvariableop^savev2_transformer_model_17_transformer_encoder_layer_normalization_1_beta_read_readvariableop<savev2_transformer_model_17_dense_kernel_read_readvariableop:savev2_transformer_model_17_dense_bias_read_readvariableop>savev2_transformer_model_17_dense_1_kernel_read_readvariableop<savev2_transformer_model_17_dense_1_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopZsavev2_adam_transformer_model_17_positional_embedding_dense_2_kernel_m_read_readvariableopXsavev2_adam_transformer_model_17_positional_embedding_dense_2_bias_m_read_readvariableoplsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_m_read_readvariableopjsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_m_read_readvariableoplsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_m_read_readvariableopqsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_m_read_readvariableoposavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_m_read_readvariableopdsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_m_read_readvariableopcsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_m_read_readvariableopXsavev2_adam_transformer_model_17_transformer_encoder_conv1d_kernel_m_read_readvariableopVsavev2_adam_transformer_model_17_transformer_encoder_conv1d_bias_m_read_readvariableopZsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_m_read_readvariableopXsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_m_read_readvariableopfsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_m_read_readvariableopesavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_m_read_readvariableopCsavev2_adam_transformer_model_17_dense_kernel_m_read_readvariableopAsavev2_adam_transformer_model_17_dense_bias_m_read_readvariableopEsavev2_adam_transformer_model_17_dense_1_kernel_m_read_readvariableopCsavev2_adam_transformer_model_17_dense_1_bias_m_read_readvariableopZsavev2_adam_transformer_model_17_positional_embedding_dense_2_kernel_v_read_readvariableopXsavev2_adam_transformer_model_17_positional_embedding_dense_2_bias_v_read_readvariableoplsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_v_read_readvariableopjsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_v_read_readvariableoplsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_v_read_readvariableopqsavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_v_read_readvariableoposavev2_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_v_read_readvariableopdsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_v_read_readvariableopcsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_v_read_readvariableopXsavev2_adam_transformer_model_17_transformer_encoder_conv1d_kernel_v_read_readvariableopVsavev2_adam_transformer_model_17_transformer_encoder_conv1d_bias_v_read_readvariableopZsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_v_read_readvariableopXsavev2_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_v_read_readvariableopfsavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_v_read_readvariableopesavev2_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_v_read_readvariableopCsavev2_adam_transformer_model_17_dense_kernel_v_read_readvariableopAsavev2_adam_transformer_model_17_dense_bias_v_read_readvariableopEsavev2_adam_transformer_model_17_dense_1_kernel_v_read_readvariableopCsavev2_adam_transformer_model_17_dense_1_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *S
dtypesI
G2E	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes๒
๏: :	:::::::::::::::
::	:: : : : : : : : : : : :	:::::::::::::::
::	::	:::::::::::::::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::*&
$
_output_shapes
::*&
$
_output_shapes
::*&
$
_output_shapes
::*&
$
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::*
&
$
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:! 

_output_shapes	
::*!&
$
_output_shapes
::*"&
$
_output_shapes
::*#&
$
_output_shapes
::*$&
$
_output_shapes
::!%

_output_shapes	
::!&

_output_shapes	
::!'

_output_shapes	
::*(&
$
_output_shapes
::!)

_output_shapes	
::**&
$
_output_shapes
::!+

_output_shapes	
::!,

_output_shapes	
::!-

_output_shapes	
::&."
 
_output_shapes
:
:!/

_output_shapes	
::%0!

_output_shapes
:	: 1

_output_shapes
::%2!

_output_shapes
:	:!3

_output_shapes	
::*4&
$
_output_shapes
::*5&
$
_output_shapes
::*6&
$
_output_shapes
::*7&
$
_output_shapes
::!8

_output_shapes	
::!9

_output_shapes	
::!:

_output_shapes	
::*;&
$
_output_shapes
::!<

_output_shapes	
::*=&
$
_output_shapes
::!>

_output_shapes	
::!?

_output_shapes	
::!@

_output_shapes	
::&A"
 
_output_shapes
:
:!B

_output_shapes	
::%C!

_output_shapes
:	: D

_output_shapes
::E

_output_shapes
: 
ภ
ช
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296289
xQ
>positional_embedding_dense_2_tensordot_readvariableop_resource:	K
<positional_embedding_dense_2_biasadd_readvariableop_resource:	
positional_embedding_296143f
Ntransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource:S
Dtransformer_encoder_multi_head_attention_add_readvariableop_resource:	\
Mtransformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource:	X
Itransformer_encoder_layer_normalization_batchnorm_readvariableop_resource:	^
Ftransformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource:I
:transformer_encoder_conv1d_biasadd_readvariableop_resource:	`
Htransformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource:K
<transformer_encoder_conv1d_1_biasadd_readvariableop_resource:	^
Otransformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource:	Z
Ktransformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identityขdense/BiasAdd/ReadVariableOpขdense/MatMul/ReadVariableOpขdense_1/BiasAdd/ReadVariableOpขdense_1/MatMul/ReadVariableOpข3positional_embedding/dense_2/BiasAdd/ReadVariableOpข5positional_embedding/dense_2/Tensordot/ReadVariableOpข1transformer_encoder/conv1d/BiasAdd/ReadVariableOpข=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpข3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpข?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpข@transformer_encoder/layer_normalization/batchnorm/ReadVariableOpขDtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpขBtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpขFtransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpข;transformer_encoder/multi_head_attention/add/ReadVariableOpขEtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpขGtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpขGtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpขGtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpK
positional_embedding/ShapeShapex*
T0*
_output_shapes
:r
(positional_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*positional_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*positional_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:บ
"positional_embedding/strided_sliceStridedSlice#positional_embedding/Shape:output:01positional_embedding/strided_slice/stack:output:03positional_embedding/strided_slice/stack_1:output:03positional_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskต
5positional_embedding/dense_2/Tensordot/ReadVariableOpReadVariableOp>positional_embedding_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0u
+positional_embedding/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+positional_embedding/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ]
,positional_embedding/dense_2/Tensordot/ShapeShapex*
T0*
_output_shapes
:v
4positional_embedding/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ฏ
/positional_embedding/dense_2/Tensordot/GatherV2GatherV25positional_embedding/dense_2/Tensordot/Shape:output:04positional_embedding/dense_2/Tensordot/free:output:0=positional_embedding/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6positional_embedding/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ณ
1positional_embedding/dense_2/Tensordot/GatherV2_1GatherV25positional_embedding/dense_2/Tensordot/Shape:output:04positional_embedding/dense_2/Tensordot/axes:output:0?positional_embedding/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,positional_embedding/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ล
+positional_embedding/dense_2/Tensordot/ProdProd8positional_embedding/dense_2/Tensordot/GatherV2:output:05positional_embedding/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.positional_embedding/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ห
-positional_embedding/dense_2/Tensordot/Prod_1Prod:positional_embedding/dense_2/Tensordot/GatherV2_1:output:07positional_embedding/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2positional_embedding/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-positional_embedding/dense_2/Tensordot/concatConcatV24positional_embedding/dense_2/Tensordot/free:output:04positional_embedding/dense_2/Tensordot/axes:output:0;positional_embedding/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ะ
,positional_embedding/dense_2/Tensordot/stackPack4positional_embedding/dense_2/Tensordot/Prod:output:06positional_embedding/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ฎ
0positional_embedding/dense_2/Tensordot/transpose	Transposex6positional_embedding/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????แ
.positional_embedding/dense_2/Tensordot/ReshapeReshape4positional_embedding/dense_2/Tensordot/transpose:y:05positional_embedding/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????โ
-positional_embedding/dense_2/Tensordot/MatMulMatMul7positional_embedding/dense_2/Tensordot/Reshape:output:0=positional_embedding/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????y
.positional_embedding/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:v
4positional_embedding/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/positional_embedding/dense_2/Tensordot/concat_1ConcatV28positional_embedding/dense_2/Tensordot/GatherV2:output:07positional_embedding/dense_2/Tensordot/Const_2:output:0=positional_embedding/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&positional_embedding/dense_2/TensordotReshape7positional_embedding/dense_2/Tensordot/MatMul:product:08positional_embedding/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????ญ
3positional_embedding/dense_2/BiasAdd/ReadVariableOpReadVariableOp<positional_embedding_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ิ
$positional_embedding/dense_2/BiasAddBiasAdd/positional_embedding/dense_2/Tensordot:output:0;positional_embedding/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????^
positional_embedding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :w
positional_embedding/CastCast$positional_embedding/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: a
positional_embedding/SqrtSqrtpositional_embedding/Cast:y:0*
T0*
_output_shapes
: ค
positional_embedding/mulMul-positional_embedding/dense_2/BiasAdd:output:0positional_embedding/Sqrt:y:0*
T0*,
_output_shapes
:?????????\
positional_embedding/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
positional_embedding/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,positional_embedding/strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : n
,positional_embedding/strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ๓
*positional_embedding/strided_slice_1/stackPack5positional_embedding/strided_slice_1/stack/0:output:0#positional_embedding/Const:output:05positional_embedding/strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:p
.positional_embedding/strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : p
.positional_embedding/strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : 
,positional_embedding/strided_slice_1/stack_1Pack7positional_embedding/strided_slice_1/stack_1/0:output:0+positional_embedding/strided_slice:output:07positional_embedding/strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:p
.positional_embedding/strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :p
.positional_embedding/strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :๛
,positional_embedding/strided_slice_1/stack_2Pack7positional_embedding/strided_slice_1/stack_2/0:output:0%positional_embedding/Const_1:output:07positional_embedding/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:ๆ
$positional_embedding/strided_slice_1StridedSlicepositional_embedding_2961433positional_embedding/strided_slice_1/stack:output:05positional_embedding/strided_slice_1/stack_1:output:05positional_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maskฅ
positional_embedding/addAddV2positional_embedding/mul:z:0-positional_embedding/strided_slice_1:output:0*
T0*,
_output_shapes
:?????????ฺ
Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
6transformer_encoder/multi_head_attention/einsum/EinsumEinsumpositional_embedding/add:z:0Mtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...NI,HIO->...NHO?
Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO?
Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_2/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHOs
.transformer_encoder/multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  C
-transformer_encoder/multi_head_attention/SqrtSqrt7transformer_encoder/multi_head_attention/Const:output:0*
T0*
_output_shapes
: ๊
0transformer_encoder/multi_head_attention/truedivRealDiv?transformer_encoder/multi_head_attention/einsum/Einsum:output:01transformer_encoder/multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:?????????ฃ
8transformer_encoder/multi_head_attention/einsum_3/EinsumEinsum4transformer_encoder/multi_head_attention/truediv:z:0Atransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:?????????*#
equation...NHO,...MHO->...HNMธ
0transformer_encoder/multi_head_attention/SoftmaxSoftmaxAtransformer_encoder/multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:?????????
@transformer_encoder/multi_head_attention/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?
>transformer_encoder/multi_head_attention/dropout_1/dropout/MulMul:transformer_encoder/multi_head_attention/Softmax:softmax:0Itransformer_encoder/multi_head_attention/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????ช
@transformer_encoder/multi_head_attention/dropout_1/dropout/ShapeShape:transformer_encoder/multi_head_attention/Softmax:softmax:0*
T0*
_output_shapes
:๚
Wtransformer_encoder/multi_head_attention/dropout_1/dropout/random_uniform/RandomUniformRandomUniformItransformer_encoder/multi_head_attention/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0
Itransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ว
Gtransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqualGreaterEqual`transformer_encoder/multi_head_attention/dropout_1/dropout/random_uniform/RandomUniform:output:0Rtransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
?transformer_encoder/multi_head_attention/dropout_1/dropout/CastCastKtransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????
@transformer_encoder/multi_head_attention/dropout_1/dropout/Mul_1MulBtransformer_encoder/multi_head_attention/dropout_1/dropout/Mul:z:0Ctransformer_encoder/multi_head_attention/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????ด
8transformer_encoder/multi_head_attention/einsum_4/EinsumEinsumDtransformer_encoder/multi_head_attention/dropout_1/dropout/Mul_1:z:0Atransformer_encoder/multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:?????????*#
equation...HNM,...MHI->...NHI?
Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ท
8transformer_encoder/multi_head_attention/einsum_5/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_4/Einsum:output:0Otransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????*
equation...NHI,HIO->...NOฝ
;transformer_encoder/multi_head_attention/add/ReadVariableOpReadVariableOpDtransformer_encoder_multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0๔
,transformer_encoder/multi_head_attention/addAddV2Atransformer_encoder/multi_head_attention/einsum_5/Einsum:output:0Ctransformer_encoder/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????p
+transformer_encoder/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?ฯ
)transformer_encoder/dropout_2/dropout/MulMul0transformer_encoder/multi_head_attention/add:z:04transformer_encoder/dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:?????????
+transformer_encoder/dropout_2/dropout/ShapeShape0transformer_encoder/multi_head_attention/add:z:0*
T0*
_output_shapes
:อ
Btransformer_encoder/dropout_2/dropout/random_uniform/RandomUniformRandomUniform4transformer_encoder/dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0y
4transformer_encoder/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>
2transformer_encoder/dropout_2/dropout/GreaterEqualGreaterEqualKtransformer_encoder/dropout_2/dropout/random_uniform/RandomUniform:output:0=transformer_encoder/dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????ฐ
*transformer_encoder/dropout_2/dropout/CastCast6transformer_encoder/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????ศ
+transformer_encoder/dropout_2/dropout/Mul_1Mul-transformer_encoder/dropout_2/dropout/Mul:z:0.transformer_encoder/dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????
Ftransformer_encoder/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
4transformer_encoder/layer_normalization/moments/meanMean/transformer_encoder/dropout_2/dropout/Mul_1:z:0Otransformer_encoder/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(ม
<transformer_encoder/layer_normalization/moments/StopGradientStopGradient=transformer_encoder/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????
Atransformer_encoder/layer_normalization/moments/SquaredDifferenceSquaredDifference/transformer_encoder/dropout_2/dropout/Mul_1:z:0Etransformer_encoder/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
Jtransformer_encoder/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ฃ
8transformer_encoder/layer_normalization/moments/varianceMeanEtransformer_encoder/layer_normalization/moments/SquaredDifference:z:0Stransformer_encoder/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(|
7transformer_encoder/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75๙
5transformer_encoder/layer_normalization/batchnorm/addAddV2Atransformer_encoder/layer_normalization/moments/variance:output:0@transformer_encoder/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????ฑ
7transformer_encoder/layer_normalization/batchnorm/RsqrtRsqrt9transformer_encoder/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ฯ
Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0?
5transformer_encoder/layer_normalization/batchnorm/mulMul;transformer_encoder/layer_normalization/batchnorm/Rsqrt:y:0Ltransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????แ
7transformer_encoder/layer_normalization/batchnorm/mul_1Mul/transformer_encoder/dropout_2/dropout/Mul_1:z:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????๏
7transformer_encoder/layer_normalization/batchnorm/mul_2Mul=transformer_encoder/layer_normalization/moments/mean:output:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ว
@transformer_encoder/layer_normalization/batchnorm/ReadVariableOpReadVariableOpItransformer_encoder_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0๚
5transformer_encoder/layer_normalization/batchnorm/subSubHtransformer_encoder/layer_normalization/batchnorm/ReadVariableOp:value:0;transformer_encoder/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????๏
7transformer_encoder/layer_normalization/batchnorm/add_1AddV2;transformer_encoder/layer_normalization/batchnorm/mul_1:z:09transformer_encoder/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ฒ
transformer_encoder/addAddV2;transformer_encoder/layer_normalization/batchnorm/add_1:z:0positional_embedding/add:z:0*
T0*,
_output_shapes
:?????????{
0transformer_encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????อ
,transformer_encoder/conv1d/Conv1D/ExpandDims
ExpandDimstransformer_encoder/add:z:09transformer_encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ส
=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpFtransformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0t
2transformer_encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ๓
.transformer_encoder/conv1d/Conv1D/ExpandDims_1
ExpandDimsEtransformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0;transformer_encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:?
!transformer_encoder/conv1d/Conv1DConv2D5transformer_encoder/conv1d/Conv1D/ExpandDims:output:07transformer_encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
ท
)transformer_encoder/conv1d/Conv1D/SqueezeSqueeze*transformer_encoder/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????ฉ
1transformer_encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp:transformer_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ำ
"transformer_encoder/conv1d/BiasAddBiasAdd2transformer_encoder/conv1d/Conv1D/Squeeze:output:09transformer_encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????
transformer_encoder/conv1d/ReluRelu+transformer_encoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????p
+transformer_encoder/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?ฬ
)transformer_encoder/dropout_3/dropout/MulMul-transformer_encoder/conv1d/Relu:activations:04transformer_encoder/dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:?????????
+transformer_encoder/dropout_3/dropout/ShapeShape-transformer_encoder/conv1d/Relu:activations:0*
T0*
_output_shapes
:อ
Btransformer_encoder/dropout_3/dropout/random_uniform/RandomUniformRandomUniform4transformer_encoder/dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0y
4transformer_encoder/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>
2transformer_encoder/dropout_3/dropout/GreaterEqualGreaterEqualKtransformer_encoder/dropout_3/dropout/random_uniform/RandomUniform:output:0=transformer_encoder/dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????ฐ
*transformer_encoder/dropout_3/dropout/CastCast6transformer_encoder/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????ศ
+transformer_encoder/dropout_3/dropout/Mul_1Mul-transformer_encoder/dropout_3/dropout/Mul:z:0.transformer_encoder/dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????}
2transformer_encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????ๅ
.transformer_encoder/conv1d_1/Conv1D/ExpandDims
ExpandDims/transformer_encoder/dropout_3/dropout/Mul_1:z:0;transformer_encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ฮ
?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpHtransformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0v
4transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ๙
0transformer_encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDimsGtransformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0=transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:
#transformer_encoder/conv1d_1/Conv1DConv2D7transformer_encoder/conv1d_1/Conv1D/ExpandDims:output:09transformer_encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
ป
+transformer_encoder/conv1d_1/Conv1D/SqueezeSqueeze,transformer_encoder/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????ญ
3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ู
$transformer_encoder/conv1d_1/BiasAddBiasAdd4transformer_encoder/conv1d_1/Conv1D/Squeeze:output:0;transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????
Htransformer_encoder/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
6transformer_encoder/layer_normalization_1/moments/meanMean-transformer_encoder/conv1d_1/BiasAdd:output:0Qtransformer_encoder/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(ล
>transformer_encoder/layer_normalization_1/moments/StopGradientStopGradient?transformer_encoder/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:?????????
Ctransformer_encoder/layer_normalization_1/moments/SquaredDifferenceSquaredDifference-transformer_encoder/conv1d_1/BiasAdd:output:0Gtransformer_encoder/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
Ltransformer_encoder/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ฉ
:transformer_encoder/layer_normalization_1/moments/varianceMeanGtransformer_encoder/layer_normalization_1/moments/SquaredDifference:z:0Utransformer_encoder/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(~
9transformer_encoder/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75?
7transformer_encoder/layer_normalization_1/batchnorm/addAddV2Ctransformer_encoder/layer_normalization_1/moments/variance:output:0Btransformer_encoder/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????ต
9transformer_encoder/layer_normalization_1/batchnorm/RsqrtRsqrt;transformer_encoder/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ำ
Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/mulMul=transformer_encoder/layer_normalization_1/batchnorm/Rsqrt:y:0Ntransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ใ
9transformer_encoder/layer_normalization_1/batchnorm/mul_1Mul-transformer_encoder/conv1d_1/BiasAdd:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????๕
9transformer_encoder/layer_normalization_1/batchnorm/mul_2Mul?transformer_encoder/layer_normalization_1/moments/mean:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ห
Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpKtransformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/subSubJtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp:value:0=transformer_encoder/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????๕
9transformer_encoder/layer_normalization_1/batchnorm/add_1AddV2=transformer_encoder/layer_normalization_1/batchnorm/mul_1:z:0;transformer_encoder/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????ต
transformer_encoder/add_1AddV2transformer_encoder/add:z:0=transformer_encoder/layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :ฑ
global_average_pooling1d/MeanMeantransformer_encoder/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:?????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:?????????]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ฟ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp4^positional_embedding/dense_2/BiasAdd/ReadVariableOp6^positional_embedding/dense_2/Tensordot/ReadVariableOp2^transformer_encoder/conv1d/BiasAdd/ReadVariableOp>^transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4^transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp@^transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpA^transformer_encoder/layer_normalization/batchnorm/ReadVariableOpE^transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpC^transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpG^transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp<^transformer_encoder/multi_head_attention/add/ReadVariableOpF^transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2j
3positional_embedding/dense_2/BiasAdd/ReadVariableOp3positional_embedding/dense_2/BiasAdd/ReadVariableOp2n
5positional_embedding/dense_2/Tensordot/ReadVariableOp5positional_embedding/dense_2/Tensordot/ReadVariableOp2f
1transformer_encoder/conv1d/BiasAdd/ReadVariableOp1transformer_encoder/conv1d/BiasAdd/ReadVariableOp2~
=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2j
3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp2
?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2
@transformer_encoder/layer_normalization/batchnorm/ReadVariableOp@transformer_encoder/layer_normalization/batchnorm/ReadVariableOp2
Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpDtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp2
Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpBtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp2
Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpFtransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp2z
;transformer_encoder/multi_head_attention/add/ReadVariableOp;transformer_encoder/multi_head_attention/add/ReadVariableOp2
Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpEtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp2
Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpGtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp2
Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpGtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp2
Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpGtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:

ใญ
ฉ=
"__inference__traced_restore_297135
file_prefix\
Iassignvariableop_transformer_model_17_positional_embedding_dense_2_kernel:	X
Iassignvariableop_1_transformer_model_17_positional_embedding_dense_2_bias:	u
]assignvariableop_2_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel:s
[assignvariableop_3_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel:u
]assignvariableop_4_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel:z
bassignvariableop_5_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel:o
`assignvariableop_6_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias:	d
Uassignvariableop_7_transformer_model_17_transformer_encoder_layer_normalization_gamma:	c
Tassignvariableop_8_transformer_model_17_transformer_encoder_layer_normalization_beta:	a
Iassignvariableop_9_transformer_model_17_transformer_encoder_conv1d_kernel:W
Hassignvariableop_10_transformer_model_17_transformer_encoder_conv1d_bias:	d
Lassignvariableop_11_transformer_model_17_transformer_encoder_conv1d_1_kernel:Y
Jassignvariableop_12_transformer_model_17_transformer_encoder_conv1d_1_bias:	g
Xassignvariableop_13_transformer_model_17_transformer_encoder_layer_normalization_1_gamma:	f
Wassignvariableop_14_transformer_model_17_transformer_encoder_layer_normalization_1_beta:	I
5assignvariableop_15_transformer_model_17_dense_kernel:
B
3assignvariableop_16_transformer_model_17_dense_bias:	J
7assignvariableop_17_transformer_model_17_dense_1_kernel:	C
5assignvariableop_18_transformer_model_17_dense_1_bias:$
assignvariableop_19_beta_1: $
assignvariableop_20_beta_2: #
assignvariableop_21_decay: +
!assignvariableop_22_learning_rate: '
assignvariableop_23_adam_iter:	 %
assignvariableop_24_total_2: %
assignvariableop_25_count_2: %
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: f
Sassignvariableop_30_adam_transformer_model_17_positional_embedding_dense_2_kernel_m:	`
Qassignvariableop_31_adam_transformer_model_17_positional_embedding_dense_2_bias_m:	}
eassignvariableop_32_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_m:{
cassignvariableop_33_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_m:}
eassignvariableop_34_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_m:
jassignvariableop_35_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_m:w
hassignvariableop_36_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_m:	l
]assignvariableop_37_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_m:	k
\assignvariableop_38_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_m:	i
Qassignvariableop_39_adam_transformer_model_17_transformer_encoder_conv1d_kernel_m:^
Oassignvariableop_40_adam_transformer_model_17_transformer_encoder_conv1d_bias_m:	k
Sassignvariableop_41_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_m:`
Qassignvariableop_42_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_m:	n
_assignvariableop_43_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_m:	m
^assignvariableop_44_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_m:	P
<assignvariableop_45_adam_transformer_model_17_dense_kernel_m:
I
:assignvariableop_46_adam_transformer_model_17_dense_bias_m:	Q
>assignvariableop_47_adam_transformer_model_17_dense_1_kernel_m:	J
<assignvariableop_48_adam_transformer_model_17_dense_1_bias_m:f
Sassignvariableop_49_adam_transformer_model_17_positional_embedding_dense_2_kernel_v:	`
Qassignvariableop_50_adam_transformer_model_17_positional_embedding_dense_2_bias_v:	}
eassignvariableop_51_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_v:{
cassignvariableop_52_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_v:}
eassignvariableop_53_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_v:
jassignvariableop_54_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_v:w
hassignvariableop_55_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_v:	l
]assignvariableop_56_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_v:	k
\assignvariableop_57_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_v:	i
Qassignvariableop_58_adam_transformer_model_17_transformer_encoder_conv1d_kernel_v:^
Oassignvariableop_59_adam_transformer_model_17_transformer_encoder_conv1d_bias_v:	k
Sassignvariableop_60_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_v:`
Qassignvariableop_61_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_v:	n
_assignvariableop_62_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_v:	m
^assignvariableop_63_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_v:	P
<assignvariableop_64_adam_transformer_model_17_dense_kernel_v:
I
:assignvariableop_65_adam_transformer_model_17_dense_bias_v:	Q
>assignvariableop_66_adam_transformer_model_17_dense_1_kernel_v:	J
<assignvariableop_67_adam_transformer_model_17_dense_1_bias_v:
identity_69ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_33ขAssignVariableOp_34ขAssignVariableOp_35ขAssignVariableOp_36ขAssignVariableOp_37ขAssignVariableOp_38ขAssignVariableOp_39ขAssignVariableOp_4ขAssignVariableOp_40ขAssignVariableOp_41ขAssignVariableOp_42ขAssignVariableOp_43ขAssignVariableOp_44ขAssignVariableOp_45ขAssignVariableOp_46ขAssignVariableOp_47ขAssignVariableOp_48ขAssignVariableOp_49ขAssignVariableOp_5ขAssignVariableOp_50ขAssignVariableOp_51ขAssignVariableOp_52ขAssignVariableOp_53ขAssignVariableOp_54ขAssignVariableOp_55ขAssignVariableOp_56ขAssignVariableOp_57ขAssignVariableOp_58ขAssignVariableOp_59ขAssignVariableOp_6ขAssignVariableOp_60ขAssignVariableOp_61ขAssignVariableOp_62ขAssignVariableOp_63ขAssignVariableOp_64ขAssignVariableOp_65ขAssignVariableOp_66ขAssignVariableOp_67ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9แ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
value?B๚EB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
valueBEB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ๚
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ช
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*S
dtypesI
G2E	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ด
AssignVariableOpAssignVariableOpIassignvariableop_transformer_model_17_positional_embedding_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ธ
AssignVariableOp_1AssignVariableOpIassignvariableop_1_transformer_model_17_positional_embedding_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ฬ
AssignVariableOp_2AssignVariableOp]assignvariableop_2_transformer_model_17_transformer_encoder_multi_head_attention_query_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ส
AssignVariableOp_3AssignVariableOp[assignvariableop_3_transformer_model_17_transformer_encoder_multi_head_attention_key_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:ฬ
AssignVariableOp_4AssignVariableOp]assignvariableop_4_transformer_model_17_transformer_encoder_multi_head_attention_value_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ั
AssignVariableOp_5AssignVariableOpbassignvariableop_5_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ฯ
AssignVariableOp_6AssignVariableOp`assignvariableop_6_transformer_model_17_transformer_encoder_multi_head_attention_projection_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ฤ
AssignVariableOp_7AssignVariableOpUassignvariableop_7_transformer_model_17_transformer_encoder_layer_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ร
AssignVariableOp_8AssignVariableOpTassignvariableop_8_transformer_model_17_transformer_encoder_layer_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ธ
AssignVariableOp_9AssignVariableOpIassignvariableop_9_transformer_model_17_transformer_encoder_conv1d_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:น
AssignVariableOp_10AssignVariableOpHassignvariableop_10_transformer_model_17_transformer_encoder_conv1d_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ฝ
AssignVariableOp_11AssignVariableOpLassignvariableop_11_transformer_model_17_transformer_encoder_conv1d_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ป
AssignVariableOp_12AssignVariableOpJassignvariableop_12_transformer_model_17_transformer_encoder_conv1d_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ษ
AssignVariableOp_13AssignVariableOpXassignvariableop_13_transformer_model_17_transformer_encoder_layer_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ศ
AssignVariableOp_14AssignVariableOpWassignvariableop_14_transformer_model_17_transformer_encoder_layer_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ฆ
AssignVariableOp_15AssignVariableOp5assignvariableop_15_transformer_model_17_dense_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ค
AssignVariableOp_16AssignVariableOp3assignvariableop_16_transformer_model_17_dense_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:จ
AssignVariableOp_17AssignVariableOp7assignvariableop_17_transformer_model_17_dense_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ฆ
AssignVariableOp_18AssignVariableOp5assignvariableop_18_transformer_model_17_dense_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp!assignvariableop_22_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ฤ
AssignVariableOp_30AssignVariableOpSassignvariableop_30_adam_transformer_model_17_positional_embedding_dense_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ย
AssignVariableOp_31AssignVariableOpQassignvariableop_31_adam_transformer_model_17_positional_embedding_dense_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ึ
AssignVariableOp_32AssignVariableOpeassignvariableop_32_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ิ
AssignVariableOp_33AssignVariableOpcassignvariableop_33_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ึ
AssignVariableOp_34AssignVariableOpeassignvariableop_34_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpjassignvariableop_35_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ู
AssignVariableOp_36AssignVariableOphassignvariableop_36_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ฮ
AssignVariableOp_37AssignVariableOp]assignvariableop_37_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:อ
AssignVariableOp_38AssignVariableOp\assignvariableop_38_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ย
AssignVariableOp_39AssignVariableOpQassignvariableop_39_adam_transformer_model_17_transformer_encoder_conv1d_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ภ
AssignVariableOp_40AssignVariableOpOassignvariableop_40_adam_transformer_model_17_transformer_encoder_conv1d_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ฤ
AssignVariableOp_41AssignVariableOpSassignvariableop_41_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ย
AssignVariableOp_42AssignVariableOpQassignvariableop_42_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:ะ
AssignVariableOp_43AssignVariableOp_assignvariableop_43_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ฯ
AssignVariableOp_44AssignVariableOp^assignvariableop_44_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:ญ
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_transformer_model_17_dense_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_46AssignVariableOp:assignvariableop_46_adam_transformer_model_17_dense_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:ฏ
AssignVariableOp_47AssignVariableOp>assignvariableop_47_adam_transformer_model_17_dense_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ญ
AssignVariableOp_48AssignVariableOp<assignvariableop_48_adam_transformer_model_17_dense_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:ฤ
AssignVariableOp_49AssignVariableOpSassignvariableop_49_adam_transformer_model_17_positional_embedding_dense_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:ย
AssignVariableOp_50AssignVariableOpQassignvariableop_50_adam_transformer_model_17_positional_embedding_dense_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:ึ
AssignVariableOp_51AssignVariableOpeassignvariableop_51_adam_transformer_model_17_transformer_encoder_multi_head_attention_query_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ิ
AssignVariableOp_52AssignVariableOpcassignvariableop_52_adam_transformer_model_17_transformer_encoder_multi_head_attention_key_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:ึ
AssignVariableOp_53AssignVariableOpeassignvariableop_53_adam_transformer_model_17_transformer_encoder_multi_head_attention_value_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpjassignvariableop_54_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:ู
AssignVariableOp_55AssignVariableOphassignvariableop_55_adam_transformer_model_17_transformer_encoder_multi_head_attention_projection_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:ฮ
AssignVariableOp_56AssignVariableOp]assignvariableop_56_adam_transformer_model_17_transformer_encoder_layer_normalization_gamma_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:อ
AssignVariableOp_57AssignVariableOp\assignvariableop_57_adam_transformer_model_17_transformer_encoder_layer_normalization_beta_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:ย
AssignVariableOp_58AssignVariableOpQassignvariableop_58_adam_transformer_model_17_transformer_encoder_conv1d_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:ภ
AssignVariableOp_59AssignVariableOpOassignvariableop_59_adam_transformer_model_17_transformer_encoder_conv1d_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ฤ
AssignVariableOp_60AssignVariableOpSassignvariableop_60_adam_transformer_model_17_transformer_encoder_conv1d_1_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ย
AssignVariableOp_61AssignVariableOpQassignvariableop_61_adam_transformer_model_17_transformer_encoder_conv1d_1_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:ะ
AssignVariableOp_62AssignVariableOp_assignvariableop_62_adam_transformer_model_17_transformer_encoder_layer_normalization_1_gamma_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:ฯ
AssignVariableOp_63AssignVariableOp^assignvariableop_63_adam_transformer_model_17_transformer_encoder_layer_normalization_1_beta_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:ญ
AssignVariableOp_64AssignVariableOp<assignvariableop_64_adam_transformer_model_17_dense_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_65AssignVariableOp:assignvariableop_65_adam_transformer_model_17_dense_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:ฏ
AssignVariableOp_66AssignVariableOp>assignvariableop_66_adam_transformer_model_17_dense_1_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:ญ
AssignVariableOp_67AssignVariableOp<assignvariableop_67_adam_transformer_model_17_dense_1_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ง
Identity_68Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_69IdentityIdentity_68:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_69Identity_69:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ฦ

(__inference_dense_1_layer_call_fn_296374

inputs
unknown:	
	unknown_0:
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_295243o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๖

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296646

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identityขconv1d/BiasAdd/ReadVariableOpข)conv1d/Conv1D/ExpandDims_1/ReadVariableOpขconv1d_1/BiasAdd/ReadVariableOpข+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpข,layer_normalization/batchnorm/ReadVariableOpข0layer_normalization/batchnorm/mul/ReadVariableOpข.layer_normalization_1/batchnorm/ReadVariableOpข2layer_normalization_1/batchnorm/mul/ReadVariableOpข'multi_head_attention/add/ReadVariableOpข1multi_head_attention/einsum/Einsum/ReadVariableOpข3multi_head_attention/einsum_1/Einsum/ReadVariableOpข3multi_head_attention/einsum_2/Einsum/ReadVariableOpข3multi_head_attention/einsum_5/Einsum/ReadVariableOpฒ
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ิ
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...NI,HIO->...NHOถ
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHOถ
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO_
multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Cg
multi_head_attention/SqrtSqrt#multi_head_attention/Const:output:0*
T0*
_output_shapes
: ฎ
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:?????????็
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:?????????*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:?????????q
,multi_head_attention/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?ส
*multi_head_attention/dropout_1/dropout/MulMul&multi_head_attention/Softmax:softmax:05multi_head_attention/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????
,multi_head_attention/dropout_1/dropout/ShapeShape&multi_head_attention/Softmax:softmax:0*
T0*
_output_shapes
:า
Cmulti_head_attention/dropout_1/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0z
5multi_head_attention/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>
3multi_head_attention/dropout_1/dropout/GreaterEqualGreaterEqualLmulti_head_attention/dropout_1/dropout/random_uniform/RandomUniform:output:0>multi_head_attention/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????ต
+multi_head_attention/dropout_1/dropout/CastCast7multi_head_attention/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????ฮ
,multi_head_attention/dropout_1/dropout/Mul_1Mul.multi_head_attention/dropout_1/dropout/Mul:z:0/multi_head_attention/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????๘
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/dropout/Mul_1:z:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:?????????*#
equation...HNM,...MHI->...NHIถ
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0๛
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0ธ
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?
dropout_2/dropout/MulMulmulti_head_attention/add:z:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:?????????c
dropout_2/dropout/ShapeShapemulti_head_attention/add:z:0*
T0*
_output_shapes
:ฅ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ษ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ษ
 layer_normalization/moments/meanMeandropout_2/dropout/Mul_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????ษ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/dropout/Mul_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:็
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ฝ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ง
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ย
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ฅ
#layer_normalization/batchnorm/mul_1Muldropout_2/dropout/Mul_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0พ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????t
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:?????????g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ข
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ท
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ร
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?
dropout_3/dropout/MulMulconv1d/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:?????????`
dropout_3/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:ฅ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ษ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????ฉ
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/dropout/Mul_1:z:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ฆ
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ฝ
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ษ
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ห
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:?????????ห
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ํ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ร
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ซ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ศ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ง
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ฃ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ฤ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????y
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????ช
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????: : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2R
'multi_head_attention/add/ReadVariableOp'multi_head_attention/add/ReadVariableOp2f
1multi_head_attention/einsum/Einsum/ReadVariableOp1multi_head_attention/einsum/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_1/Einsum/ReadVariableOp3multi_head_attention/einsum_1/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_2/Einsum/ReadVariableOp3multi_head_attention/einsum_2/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_5/Einsum/ReadVariableOp3multi_head_attention/einsum_5/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
๖

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295480

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identityขconv1d/BiasAdd/ReadVariableOpข)conv1d/Conv1D/ExpandDims_1/ReadVariableOpขconv1d_1/BiasAdd/ReadVariableOpข+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpข,layer_normalization/batchnorm/ReadVariableOpข0layer_normalization/batchnorm/mul/ReadVariableOpข.layer_normalization_1/batchnorm/ReadVariableOpข2layer_normalization_1/batchnorm/mul/ReadVariableOpข'multi_head_attention/add/ReadVariableOpข1multi_head_attention/einsum/Einsum/ReadVariableOpข3multi_head_attention/einsum_1/Einsum/ReadVariableOpข3multi_head_attention/einsum_2/Einsum/ReadVariableOpข3multi_head_attention/einsum_5/Einsum/ReadVariableOpฒ
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ิ
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...NI,HIO->...NHOถ
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHOถ
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO_
multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Cg
multi_head_attention/SqrtSqrt#multi_head_attention/Const:output:0*
T0*
_output_shapes
: ฎ
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:?????????็
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:?????????*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:?????????q
,multi_head_attention/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?ส
*multi_head_attention/dropout_1/dropout/MulMul&multi_head_attention/Softmax:softmax:05multi_head_attention/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????
,multi_head_attention/dropout_1/dropout/ShapeShape&multi_head_attention/Softmax:softmax:0*
T0*
_output_shapes
:า
Cmulti_head_attention/dropout_1/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0z
5multi_head_attention/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>
3multi_head_attention/dropout_1/dropout/GreaterEqualGreaterEqualLmulti_head_attention/dropout_1/dropout/random_uniform/RandomUniform:output:0>multi_head_attention/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????ต
+multi_head_attention/dropout_1/dropout/CastCast7multi_head_attention/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????ฮ
,multi_head_attention/dropout_1/dropout/Mul_1Mul.multi_head_attention/dropout_1/dropout/Mul:z:0/multi_head_attention/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????๘
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/dropout/Mul_1:z:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:?????????*#
equation...HNM,...MHI->...NHIถ
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0๛
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0ธ
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?
dropout_2/dropout/MulMulmulti_head_attention/add:z:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:?????????c
dropout_2/dropout/ShapeShapemulti_head_attention/add:z:0*
T0*
_output_shapes
:ฅ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ษ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ษ
 layer_normalization/moments/meanMeandropout_2/dropout/Mul_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????ษ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/dropout/Mul_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:็
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ฝ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ง
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ย
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ฅ
#layer_normalization/batchnorm/mul_1Muldropout_2/dropout/Mul_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0พ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????t
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:?????????g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ข
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ท
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ร
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n?ถ?
dropout_3/dropout/MulMulconv1d/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:?????????`
dropout_3/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:ฅ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ษ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????ฉ
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/dropout/Mul_1:z:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ฆ
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ฝ
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ษ
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ห
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:?????????ห
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ํ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ร
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ซ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ศ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ง
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ฃ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ฤ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????y
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????ช
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????: : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2R
'multi_head_attention/add/ReadVariableOp'multi_head_attention/add/ReadVariableOp2f
1multi_head_attention/einsum/Einsum/ReadVariableOp1multi_head_attention/einsum/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_1/Einsum/ReadVariableOp3multi_head_attention/einsum_1/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_2/Einsum/ReadVariableOp3multi_head_attention/einsum_2/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_5/Einsum/ReadVariableOp3multi_head_attention/einsum_5/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
๊
๕
4__inference_transformer_encoder_layer_call_fn_296416

inputs
unknown:!
	unknown_0:!
	unknown_1:!
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	!
	unknown_6:
	unknown_7:	!
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	
identityขStatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295179t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ฮ'
	
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295806
input_1.
positional_embedding_295759:	*
positional_embedding_295761:	
positional_embedding_2957632
transformer_encoder_295766:2
transformer_encoder_295768:2
transformer_encoder_295770:2
transformer_encoder_295772:)
transformer_encoder_295774:	)
transformer_encoder_295776:	)
transformer_encoder_295778:	2
transformer_encoder_295780:)
transformer_encoder_295782:	2
transformer_encoder_295784:)
transformer_encoder_295786:	)
transformer_encoder_295788:	)
transformer_encoder_295790:	 
dense_295794:

dense_295796:	!
dense_1_295800:	
dense_1_295802:
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdropout/StatefulPartitionedCallข,positional_embedding/StatefulPartitionedCallข+transformer_encoder/StatefulPartitionedCallว
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1positional_embedding_295759positional_embedding_295761positional_embedding_295763*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_295082
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_295766transformer_encoder_295768transformer_encoder_295770transformer_encoder_295772transformer_encoder_295774transformer_encoder_295776transformer_encoder_295778transformer_encoder_295780transformer_encoder_295782transformer_encoder_295784transformer_encoder_295786transformer_encoder_295788transformer_encoder_295790*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295480
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_295018
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_295794dense_295796*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_295219๊
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_295323
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_295800dense_1_295802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_295243w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1:&"
 
_output_shapes
:

ฃ

๕
C__inference_dense_1_layer_call_and_return_conditional_losses_295243

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
?
U
9__inference_global_average_pooling1d_layer_call_fn_296359

inputs
identityห
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_295018i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
&
๋
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295250
x.
positional_embedding_295083:	*
positional_embedding_295085:	
positional_embedding_2950872
transformer_encoder_295180:2
transformer_encoder_295182:2
transformer_encoder_295184:2
transformer_encoder_295186:)
transformer_encoder_295188:	)
transformer_encoder_295190:	)
transformer_encoder_295192:	2
transformer_encoder_295194:)
transformer_encoder_295196:	2
transformer_encoder_295198:)
transformer_encoder_295200:	)
transformer_encoder_295202:	)
transformer_encoder_295204:	 
dense_295220:

dense_295222:	!
dense_1_295244:	
dense_1_295246:
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallข,positional_embedding/StatefulPartitionedCallข+transformer_encoder/StatefulPartitionedCallม
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallxpositional_embedding_295083positional_embedding_295085positional_embedding_295087*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_295082
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_295180transformer_encoder_295182transformer_encoder_295184transformer_encoder_295186transformer_encoder_295188transformer_encoder_295190transformer_encoder_295192transformer_encoder_295194transformer_encoder_295196transformer_encoder_295198transformer_encoder_295200transformer_encoder_295202transformer_encoder_295204*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295179
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_295018
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_295220dense_295222*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_295219ฺ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_295230
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_295244dense_1_295246*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_295243w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ๅ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:

ฎ&
๑
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295756
input_1.
positional_embedding_295709:	*
positional_embedding_295711:	
positional_embedding_2957132
transformer_encoder_295716:2
transformer_encoder_295718:2
transformer_encoder_295720:2
transformer_encoder_295722:)
transformer_encoder_295724:	)
transformer_encoder_295726:	)
transformer_encoder_295728:	2
transformer_encoder_295730:)
transformer_encoder_295732:	2
transformer_encoder_295734:)
transformer_encoder_295736:	)
transformer_encoder_295738:	)
transformer_encoder_295740:	 
dense_295744:

dense_295746:	!
dense_1_295750:	
dense_1_295752:
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallข,positional_embedding/StatefulPartitionedCallข+transformer_encoder/StatefulPartitionedCallว
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1positional_embedding_295709positional_embedding_295711positional_embedding_295713*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_295082
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_295716transformer_encoder_295718transformer_encoder_295720transformer_encoder_295722transformer_encoder_295724transformer_encoder_295726transformer_encoder_295728transformer_encoder_295730transformer_encoder_295732transformer_encoder_295734transformer_encoder_295736transformer_encoder_295738transformer_encoder_295740*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_295179
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_295018
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_295744dense_295746*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_295219ฺ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_295230
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_295750dense_1_295752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_295243w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ๅ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1:&"
 
_output_shapes
:

ฒ|

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296536

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identityขconv1d/BiasAdd/ReadVariableOpข)conv1d/Conv1D/ExpandDims_1/ReadVariableOpขconv1d_1/BiasAdd/ReadVariableOpข+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpข,layer_normalization/batchnorm/ReadVariableOpข0layer_normalization/batchnorm/mul/ReadVariableOpข.layer_normalization_1/batchnorm/ReadVariableOpข2layer_normalization_1/batchnorm/mul/ReadVariableOpข'multi_head_attention/add/ReadVariableOpข1multi_head_attention/einsum/Einsum/ReadVariableOpข3multi_head_attention/einsum_1/Einsum/ReadVariableOpข3multi_head_attention/einsum_2/Einsum/ReadVariableOpข3multi_head_attention/einsum_5/Einsum/ReadVariableOpฒ
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ิ
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...NI,HIO->...NHOถ
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHOถ
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ุ
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:?????????*
equation...MI,HIO->...MHO_
multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  Cg
multi_head_attention/SqrtSqrt#multi_head_attention/Const:output:0*
T0*
_output_shapes
: ฎ
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:?????????็
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:?????????*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:?????????
'multi_head_attention/dropout_1/IdentityIdentity&multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:?????????๘
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/Identity:output:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:?????????*#
equation...HNM,...MHI->...NHIถ
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0๛
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:?????????*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0ธ
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????s
dropout_2/IdentityIdentitymulti_head_attention/add:z:0*
T0*,
_output_shapes
:?????????|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ษ
 layer_normalization/moments/meanMeandropout_2/Identity:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:?????????ษ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/Identity:output:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:็
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ฝ
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ง
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ย
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ฅ
#layer_normalization/batchnorm/mul_1Muldropout_2/Identity:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0พ
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????ณ
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????t
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:?????????g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ข
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ท
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ร
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????p
dropout_3/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:?????????i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????ฉ
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ฆ
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ฝ
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ษ
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

?????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ห
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:?????????ห
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ํ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ฝ75ร
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????ซ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ศ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????ง
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????ฃ
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ฤ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:?????????น
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????y
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:?????????ช
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????: : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2R
'multi_head_attention/add/ReadVariableOp'multi_head_attention/add/ReadVariableOp2f
1multi_head_attention/einsum/Einsum/ReadVariableOp1multi_head_attention/einsum/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_1/Einsum/ReadVariableOp3multi_head_attention/einsum_1/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_2/Einsum/ReadVariableOp3multi_head_attention/einsum_2/Einsum/ReadVariableOp2j
3multi_head_attention/einsum_5/Einsum/ReadVariableOp3multi_head_attention/einsum_5/Einsum/ReadVariableOp:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ฑ
ฎ
5__inference_positional_embedding_layer_call_fn_296300
x
unknown:	
	unknown_0:	
	unknown_1
identityขStatefulPartitionedCall๔
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_295082t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : :
22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:


p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_296365

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ค

๕
A__inference_dense_layer_call_and_return_conditional_losses_296666

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ฐ
ฏ
5__inference_transformer_model_17_layer_call_fn_295904
x
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	!
	unknown_9:

unknown_10:	"

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	

unknown_18:
identityขStatefulPartitionedCallี
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295250o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:?????????: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex:&"
 
_output_shapes
:
"ฟL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ฏ
serving_default
?
input_14
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ด
ฦ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	mlp_units
		optimizer

positional_embedding
encoders
avg_pool

mlp_layers

mlp_output

signatures"
_tf_keras_model
ฎ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18"
trackable_list_wrapper
ฎ
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18"
trackable_list_wrapper
 "
trackable_list_wrapper
ส
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
๘
(trace_0
)trace_1
*trace_2
+trace_32
5__inference_transformer_model_17_layer_call_fn_295293
5__inference_transformer_model_17_layer_call_fn_295904
5__inference_transformer_model_17_layer_call_fn_295949
5__inference_transformer_model_17_layer_call_fn_295706ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z(trace_0z)trace_1z*trace_2z+trace_3
ไ
,trace_0
-trace_1
.trace_2
/trace_32๙
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296105
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296289
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295756
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295806ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z,trace_0z-trace_1z.trace_2z/trace_3
ฬBษ
!__inference__wrapped_model_295008input_1"
ฒ
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
ฯ

0beta_1

1beta_2
	2decay
3learning_rate
4itermmmmmmmmmmmmmmmm? mก!mข"mฃvคvฅvฆvงvจvฉvชvซvฌvญvฎvฏvฐvฑvฒvณ vด!vต"vถ"
	optimizer
ด
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;	embedding"
_tf_keras_layer
'
<0"
trackable_list_wrapper
ฅ
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
.
C0
D1"
trackable_list_wrapper
ป
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
,
Kserving_default"
signature_map
K:I	28transformer_model_17/positional_embedding/dense_2/kernel
E:C26transformer_model_17/positional_embedding/dense_2/bias
b:`2Jtransformer_model_17/transformer_encoder/multi_head_attention/query_kernel
`:^2Htransformer_model_17/transformer_encoder/multi_head_attention/key_kernel
b:`2Jtransformer_model_17/transformer_encoder/multi_head_attention/value_kernel
g:e2Otransformer_model_17/transformer_encoder/multi_head_attention/projection_kernel
\:Z2Mtransformer_model_17/transformer_encoder/multi_head_attention/projection_bias
Q:O2Btransformer_model_17/transformer_encoder/layer_normalization/gamma
P:N2Atransformer_model_17/transformer_encoder/layer_normalization/beta
N:L26transformer_model_17/transformer_encoder/conv1d/kernel
C:A24transformer_model_17/transformer_encoder/conv1d/bias
P:N28transformer_model_17/transformer_encoder/conv1d_1/kernel
E:C26transformer_model_17/transformer_encoder/conv1d_1/bias
S:Q2Dtransformer_model_17/transformer_encoder/layer_normalization_1/gamma
R:P2Ctransformer_model_17/transformer_encoder/layer_normalization_1/beta
5:3
2!transformer_model_17/dense/kernel
.:,2transformer_model_17/dense/bias
6:4	2#transformer_model_17/dense_1/kernel
/:-2!transformer_model_17/dense_1/bias
 "
trackable_list_wrapper
J

0
<1
2
C3
D4
5"
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
๖B๓
5__inference_transformer_model_17_layer_call_fn_295293input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๐Bํ
5__inference_transformer_model_17_layer_call_fn_295904x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๐Bํ
5__inference_transformer_model_17_layer_call_fn_295949x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๖B๓
5__inference_transformer_model_17_layer_call_fn_295706input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296105x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296289x"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295756input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295806input_1"ฎ
ฅฒก
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
๔
Ttrace_02ื
5__inference_positional_embedding_layer_call_fn_296300
ฒ
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zTtrace_0

Utrace_02๒
P__inference_positional_embedding_layer_call_and_return_conditional_losses_296354
ฒ
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zUtrace_0
ป
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer

\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b
attn_heads
c
attn_multi
dattn_dropout
e	attn_norm
fff_conv1
g
ff_dropout
hff_conv2
iff_norm"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object

otrace_02ํ
9__inference_global_average_pooling1d_layer_call_fn_296359ฏ
ฆฒข
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsข

 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zotrace_0
ฅ
ptrace_02
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_296365ฏ
ฆฒข
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsข

 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zptrace_0
ป
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
ผ
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses
}_random_generator"
_tf_keras_layer
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
๎
trace_02ฯ
(__inference_dense_1_layer_call_fn_296374ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02๊
C__inference_dense_1_layer_call_and_return_conditional_losses_296385ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
หBศ
$__inference_signature_wrapper_295859input_1"
ฒ
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
฿B?
5__inference_positional_embedding_layer_call_fn_296300x"
ฒ
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๚B๗
P__inference_positional_embedding_layer_call_and_return_conditional_losses_296354x"
ฒ
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
?
trace_0
trace_12ข
4__inference_transformer_encoder_layer_call_fn_296416
4__inference_transformer_encoder_layer_call_fn_296447ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0ztrace_1

trace_0
?trace_12ุ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296536
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296646ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0z?trace_1
 "
trackable_list_wrapper

ก	variables
ขtrainable_variables
ฃregularization_losses
ค	keras_api
ฅ__call__
+ฆ&call_and_return_all_conditional_losses
งdropout
query_kernel

key_kernel
value_kernel
projection_kernel
projection_bias"
_tf_keras_layer
ร
จ	variables
ฉtrainable_variables
ชregularization_losses
ซ	keras_api
ฌ__call__
+ญ&call_and_return_all_conditional_losses
ฎ_random_generator"
_tf_keras_layer
ห
ฏ	variables
ฐtrainable_variables
ฑregularization_losses
ฒ	keras_api
ณ__call__
+ด&call_and_return_all_conditional_losses
	ตaxis
	gamma
beta"
_tf_keras_layer
ไ
ถ	variables
ทtrainable_variables
ธregularization_losses
น	keras_api
บ__call__
+ป&call_and_return_all_conditional_losses

kernel
bias
!ผ_jit_compiled_convolution_op"
_tf_keras_layer
ร
ฝ	variables
พtrainable_variables
ฟregularization_losses
ภ	keras_api
ม__call__
+ย&call_and_return_all_conditional_losses
ร_random_generator"
_tf_keras_layer
ไ
ฤ	variables
ลtrainable_variables
ฦregularization_losses
ว	keras_api
ศ__call__
+ษ&call_and_return_all_conditional_losses

kernel
bias
!ส_jit_compiled_convolution_op"
_tf_keras_layer
ห
ห	variables
ฬtrainable_variables
อregularization_losses
ฮ	keras_api
ฯ__call__
+ะ&call_and_return_all_conditional_losses
	ัaxis
	gamma
beta"
_tf_keras_layer
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
๚B๗
9__inference_global_average_pooling1d_layer_call_fn_296359inputs"ฏ
ฆฒข
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsข

 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_296365inputs"ฏ
ฆฒข
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsข

 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
าnon_trainable_variables
ำlayers
ิmetrics
 ีlayer_regularization_losses
ึlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
์
ืtrace_02อ
&__inference_dense_layer_call_fn_296655ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zืtrace_0

ุtrace_02่
A__inference_dense_layer_call_and_return_conditional_losses_296666ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zุtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
ูnon_trainable_variables
ฺlayers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
ฦ
?trace_0
฿trace_12
(__inference_dropout_layer_call_fn_296671
(__inference_dropout_layer_call_fn_296676ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z?trace_0z฿trace_1
?
เtrace_0
แtrace_12ม
C__inference_dropout_layer_call_and_return_conditional_losses_296681
C__inference_dropout_layer_call_and_return_conditional_losses_296693ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zเtrace_0zแtrace_1
"
_generic_user_object
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
?Bู
(__inference_dense_1_layer_call_fn_296374inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๗B๔
C__inference_dense_1_layer_call_and_return_conditional_losses_296385inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
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
Q
c0
d1
e2
f3
g4
h5
i6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
๙B๖
4__inference_transformer_encoder_layer_call_fn_296416inputs"ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๙B๖
4__inference_transformer_encoder_layer_call_fn_296447inputs"ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296536inputs"ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
B
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296646inputs"ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
โnon_trainable_variables
ใlayers
ไmetrics
 ๅlayer_regularization_losses
ๆlayer_metrics
ก	variables
ขtrainable_variables
ฃregularization_losses
ฅ__call__
+ฆ&call_and_return_all_conditional_losses
'ฆ"call_and_return_conditional_losses"
_generic_user_object
ฦ2รภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฦ2รภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ร
็	variables
่trainable_variables
้regularization_losses
๊	keras_api
๋__call__
+์&call_and_return_all_conditional_losses
ํ_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๎non_trainable_variables
๏layers
๐metrics
 ๑layer_regularization_losses
๒layer_metrics
จ	variables
ฉtrainable_variables
ชregularization_losses
ฌ__call__
+ญ&call_and_return_all_conditional_losses
'ญ"call_and_return_conditional_losses"
_generic_user_object
บ2ทด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
บ2ทด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๓non_trainable_variables
๔layers
๕metrics
 ๖layer_regularization_losses
๗layer_metrics
ฏ	variables
ฐtrainable_variables
ฑregularization_losses
ณ__call__
+ด&call_and_return_all_conditional_losses
'ด"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๘non_trainable_variables
๙layers
๚metrics
 ๛layer_regularization_losses
?layer_metrics
ถ	variables
ทtrainable_variables
ธregularization_losses
บ__call__
+ป&call_and_return_all_conditional_losses
'ป"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
?non_trainable_variables
?layers
?metrics
 layer_regularization_losses
layer_metrics
ฝ	variables
พtrainable_variables
ฟregularization_losses
ม__call__
+ย&call_and_return_all_conditional_losses
'ย"call_and_return_conditional_losses"
_generic_user_object
บ2ทด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
บ2ทด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ฤ	variables
ลtrainable_variables
ฦregularization_losses
ศ__call__
+ษ&call_and_return_all_conditional_losses
'ษ"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ห	variables
ฬtrainable_variables
อregularization_losses
ฯ__call__
+ะ&call_and_return_all_conditional_losses
'ะ"call_and_return_conditional_losses"
_generic_user_object
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
ฺBื
&__inference_dense_layer_call_fn_296655inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๕B๒
A__inference_dense_layer_call_and_return_conditional_losses_296666inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
๎B๋
(__inference_dropout_layer_call_fn_296671inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๎B๋
(__inference_dropout_layer_call_fn_296676inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_296681inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_296693inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
 "
trackable_list_wrapper
(
ง0"
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
ธ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
็	variables
่trainable_variables
้regularization_losses
๋__call__
+์&call_and_return_all_conditional_losses
'์"call_and_return_conditional_losses"
_generic_user_object
บ2ทด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
บ2ทด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
"
_generic_user_object
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
P:N	2?Adam/transformer_model_17/positional_embedding/dense_2/kernel/m
J:H2=Adam/transformer_model_17/positional_embedding/dense_2/bias/m
g:e2QAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/m
e:c2OAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/m
g:e2QAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/m
l:j2VAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/m
a:_2TAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/m
V:T2IAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/m
U:S2HAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/m
S:Q2=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/m
H:F2;Adam/transformer_model_17/transformer_encoder/conv1d/bias/m
U:S2?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/m
J:H2=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/m
X:V2KAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/m
W:U2JAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/m
::8
2(Adam/transformer_model_17/dense/kernel/m
3:12&Adam/transformer_model_17/dense/bias/m
;:9	2*Adam/transformer_model_17/dense_1/kernel/m
4:22(Adam/transformer_model_17/dense_1/bias/m
P:N	2?Adam/transformer_model_17/positional_embedding/dense_2/kernel/v
J:H2=Adam/transformer_model_17/positional_embedding/dense_2/bias/v
g:e2QAdam/transformer_model_17/transformer_encoder/multi_head_attention/query_kernel/v
e:c2OAdam/transformer_model_17/transformer_encoder/multi_head_attention/key_kernel/v
g:e2QAdam/transformer_model_17/transformer_encoder/multi_head_attention/value_kernel/v
l:j2VAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_kernel/v
a:_2TAdam/transformer_model_17/transformer_encoder/multi_head_attention/projection_bias/v
V:T2IAdam/transformer_model_17/transformer_encoder/layer_normalization/gamma/v
U:S2HAdam/transformer_model_17/transformer_encoder/layer_normalization/beta/v
S:Q2=Adam/transformer_model_17/transformer_encoder/conv1d/kernel/v
H:F2;Adam/transformer_model_17/transformer_encoder/conv1d/bias/v
U:S2?Adam/transformer_model_17/transformer_encoder/conv1d_1/kernel/v
J:H2=Adam/transformer_model_17/transformer_encoder/conv1d_1/bias/v
X:V2KAdam/transformer_model_17/transformer_encoder/layer_normalization_1/gamma/v
W:U2JAdam/transformer_model_17/transformer_encoder/layer_normalization_1/beta/v
::8
2(Adam/transformer_model_17/dense/kernel/v
3:12&Adam/transformer_model_17/dense/bias/v
;:9	2*Adam/transformer_model_17/dense_1/kernel/v
4:22(Adam/transformer_model_17/dense_1/bias/v
J
Constjtf.TrackableConstantจ
!__inference__wrapped_model_295008ท !"4ข1
*ข'
%"
input_1?????????
ช "3ช0
.
output_1"
output_1?????????ค
C__inference_dense_1_layer_call_and_return_conditional_losses_296385]!"0ข-
&ข#
!
inputs?????????
ช "%ข"

0?????????
 |
(__inference_dense_1_layer_call_fn_296374P!"0ข-
&ข#
!
inputs?????????
ช "?????????ฃ
A__inference_dense_layer_call_and_return_conditional_losses_296666^ 0ข-
&ข#
!
inputs?????????
ช "&ข#

0?????????
 {
&__inference_dense_layer_call_fn_296655Q 0ข-
&ข#
!
inputs?????????
ช "?????????ฅ
C__inference_dropout_layer_call_and_return_conditional_losses_296681^4ข1
*ข'
!
inputs?????????
p 
ช "&ข#

0?????????
 ฅ
C__inference_dropout_layer_call_and_return_conditional_losses_296693^4ข1
*ข'
!
inputs?????????
p
ช "&ข#

0?????????
 }
(__inference_dropout_layer_call_fn_296671Q4ข1
*ข'
!
inputs?????????
p 
ช "?????????}
(__inference_dropout_layer_call_fn_296676Q4ข1
*ข'
!
inputs?????????
p
ช "?????????ำ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_296365{IขF
?ข<
63
inputs'???????????????????????????

 
ช ".ข+
$!
0??????????????????
 ซ
9__inference_global_average_pooling1d_layer_call_fn_296359nIขF
?ข<
63
inputs'???????????????????????????

 
ช "!??????????????????ถ
P__inference_positional_embedding_layer_call_and_return_conditional_losses_296354bท.ข+
$ข!

x?????????
ช "*ข'
 
0?????????
 
5__inference_positional_embedding_layer_call_fn_296300Uท.ข+
$ข!

x?????????
ช "?????????ถ
$__inference_signature_wrapper_295859ท !"?ข<
ข 
5ช2
0
input_1%"
input_1?????????"3ช0
.
output_1"
output_1?????????ศ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296536u8ข5
.ข+
%"
inputs?????????
p 
ช "*ข'
 
0?????????
 ศ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_296646u8ข5
.ข+
%"
inputs?????????
p
ช "*ข'
 
0?????????
 ?
4__inference_transformer_encoder_layer_call_fn_296416h8ข5
.ข+
%"
inputs?????????
p 
ช "??????????
4__inference_transformer_encoder_layer_call_fn_296447h8ข5
.ข+
%"
inputs?????????
p
ช "?????????ฬ
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295756xท !"8ข5
.ข+
%"
input_1?????????
p 
ช "%ข"

0?????????
 ฬ
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_295806xท !"8ข5
.ข+
%"
input_1?????????
p
ช "%ข"

0?????????
 ฦ
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296105rท !"2ข/
(ข%

x?????????
p 
ช "%ข"

0?????????
 ฦ
P__inference_transformer_model_17_layer_call_and_return_conditional_losses_296289rท !"2ข/
(ข%

x?????????
p
ช "%ข"

0?????????
 ค
5__inference_transformer_model_17_layer_call_fn_295293kท !"8ข5
.ข+
%"
input_1?????????
p 
ช "?????????ค
5__inference_transformer_model_17_layer_call_fn_295706kท !"8ข5
.ข+
%"
input_1?????????
p
ช "?????????
5__inference_transformer_model_17_layer_call_fn_295904eท !"2ข/
(ข%

x?????????
p 
ช "?????????
5__inference_transformer_model_17_layer_call_fn_295949eท !"2ข/
(ข%

x?????????
p
ช "?????????