²µ(
á
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
®
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
Á
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
executor_typestring ¨
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
÷
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
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018à%
¦
'Adam/transformer_model_3/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/transformer_model_3/dense_1/bias/v

;Adam/transformer_model_3/dense_1/bias/v/Read/ReadVariableOpReadVariableOp'Adam/transformer_model_3/dense_1/bias/v*
_output_shapes
:*
dtype0
¯
)Adam/transformer_model_3/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*:
shared_name+)Adam/transformer_model_3/dense_1/kernel/v
¨
=Adam/transformer_model_3/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/transformer_model_3/dense_1/kernel/v*
_output_shapes
:	*
dtype0
£
%Adam/transformer_model_3/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/transformer_model_3/dense/bias/v

9Adam/transformer_model_3/dense/bias/v/Read/ReadVariableOpReadVariableOp%Adam/transformer_model_3/dense/bias/v*
_output_shapes	
:*
dtype0
¬
'Adam/transformer_model_3/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/transformer_model_3/dense/kernel/v
¥
;Adam/transformer_model_3/dense/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/transformer_model_3/dense/kernel/v* 
_output_shapes
:
*
dtype0
ë
IAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/v
ä
]Adam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/v*
_output_shapes	
:*
dtype0
í
JAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/v
æ
^Adam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOpJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/v*
_output_shapes	
:*
dtype0
Ñ
<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/v
Ê
PAdam/transformer_model_3/transformer_encoder/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/v*
_output_shapes	
:*
dtype0
Þ
>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/v
×
RAdam/transformer_model_3/transformer_encoder/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/v*$
_output_shapes
:*
dtype0
Í
:Adam/transformer_model_3/transformer_encoder/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_model_3/transformer_encoder/conv1d/bias/v
Æ
NAdam/transformer_model_3/transformer_encoder/conv1d/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_model_3/transformer_encoder/conv1d/bias/v*
_output_shapes	
:*
dtype0
Ú
<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_model_3/transformer_encoder/conv1d/kernel/v
Ó
PAdam/transformer_model_3/transformer_encoder/conv1d/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/v*$
_output_shapes
:*
dtype0
ç
GAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/v
à
[Adam/transformer_model_3/transformer_encoder/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/v*
_output_shapes	
:*
dtype0
é
HAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/v
â
\Adam/transformer_model_3/transformer_encoder/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOpHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/v*
_output_shapes	
:*
dtype0
ÿ
SAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*d
shared_nameUSAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/v
ø
gAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/v/Read/ReadVariableOpReadVariableOpSAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/v*
_output_shapes	
:*
dtype0

UAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*f
shared_nameWUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/v

iAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/v/Read/ReadVariableOpReadVariableOpUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/v*$
_output_shapes
:*
dtype0

PAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/v
û
dAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/v/Read/ReadVariableOpReadVariableOpPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/v*$
_output_shapes
:*
dtype0
þ
NAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*_
shared_namePNAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/v
÷
bAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/v/Read/ReadVariableOpReadVariableOpNAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/v*$
_output_shapes
:*
dtype0

PAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/v
û
dAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/v/Read/ReadVariableOpReadVariableOpPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/v*$
_output_shapes
:*
dtype0
Ñ
<Adam/transformer_model_3/positional_embedding/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_model_3/positional_embedding/dense_2/bias/v
Ê
PAdam/transformer_model_3/positional_embedding/dense_2/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_model_3/positional_embedding/dense_2/bias/v*
_output_shapes	
:*
dtype0
Ù
>Adam/transformer_model_3/positional_embedding/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>Adam/transformer_model_3/positional_embedding/dense_2/kernel/v
Ò
RAdam/transformer_model_3/positional_embedding/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_model_3/positional_embedding/dense_2/kernel/v*
_output_shapes
:	*
dtype0
¦
'Adam/transformer_model_3/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/transformer_model_3/dense_1/bias/m

;Adam/transformer_model_3/dense_1/bias/m/Read/ReadVariableOpReadVariableOp'Adam/transformer_model_3/dense_1/bias/m*
_output_shapes
:*
dtype0
¯
)Adam/transformer_model_3/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*:
shared_name+)Adam/transformer_model_3/dense_1/kernel/m
¨
=Adam/transformer_model_3/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/transformer_model_3/dense_1/kernel/m*
_output_shapes
:	*
dtype0
£
%Adam/transformer_model_3/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/transformer_model_3/dense/bias/m

9Adam/transformer_model_3/dense/bias/m/Read/ReadVariableOpReadVariableOp%Adam/transformer_model_3/dense/bias/m*
_output_shapes	
:*
dtype0
¬
'Adam/transformer_model_3/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'Adam/transformer_model_3/dense/kernel/m
¥
;Adam/transformer_model_3/dense/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/transformer_model_3/dense/kernel/m* 
_output_shapes
:
*
dtype0
ë
IAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/m
ä
]Adam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/m*
_output_shapes	
:*
dtype0
í
JAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/m
æ
^Adam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOpJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/m*
_output_shapes	
:*
dtype0
Ñ
<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/m
Ê
PAdam/transformer_model_3/transformer_encoder/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/m*
_output_shapes	
:*
dtype0
Þ
>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/m
×
RAdam/transformer_model_3/transformer_encoder/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/m*$
_output_shapes
:*
dtype0
Í
:Adam/transformer_model_3/transformer_encoder/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:Adam/transformer_model_3/transformer_encoder/conv1d/bias/m
Æ
NAdam/transformer_model_3/transformer_encoder/conv1d/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_model_3/transformer_encoder/conv1d/bias/m*
_output_shapes	
:*
dtype0
Ú
<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_model_3/transformer_encoder/conv1d/kernel/m
Ó
PAdam/transformer_model_3/transformer_encoder/conv1d/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/m*$
_output_shapes
:*
dtype0
ç
GAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/m
à
[Adam/transformer_model_3/transformer_encoder/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/m*
_output_shapes	
:*
dtype0
é
HAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/m
â
\Adam/transformer_model_3/transformer_encoder/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOpHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/m*
_output_shapes	
:*
dtype0
ÿ
SAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*d
shared_nameUSAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/m
ø
gAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/m/Read/ReadVariableOpReadVariableOpSAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/m*
_output_shapes	
:*
dtype0

UAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*f
shared_nameWUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/m

iAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/m/Read/ReadVariableOpReadVariableOpUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/m*$
_output_shapes
:*
dtype0

PAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/m
û
dAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/m/Read/ReadVariableOpReadVariableOpPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/m*$
_output_shapes
:*
dtype0
þ
NAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*_
shared_namePNAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/m
÷
bAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/m/Read/ReadVariableOpReadVariableOpNAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/m*$
_output_shapes
:*
dtype0

PAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*a
shared_nameRPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/m
û
dAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/m/Read/ReadVariableOpReadVariableOpPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/m*$
_output_shapes
:*
dtype0
Ñ
<Adam/transformer_model_3/positional_embedding/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_model_3/positional_embedding/dense_2/bias/m
Ê
PAdam/transformer_model_3/positional_embedding/dense_2/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_model_3/positional_embedding/dense_2/bias/m*
_output_shapes	
:*
dtype0
Ù
>Adam/transformer_model_3/positional_embedding/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*O
shared_name@>Adam/transformer_model_3/positional_embedding/dense_2/kernel/m
Ò
RAdam/transformer_model_3/positional_embedding/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_model_3/positional_embedding/dense_2/kernel/m*
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

 transformer_model_3/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" transformer_model_3/dense_1/bias

4transformer_model_3/dense_1/bias/Read/ReadVariableOpReadVariableOp transformer_model_3/dense_1/bias*
_output_shapes
:*
dtype0
¡
"transformer_model_3/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"transformer_model_3/dense_1/kernel

6transformer_model_3/dense_1/kernel/Read/ReadVariableOpReadVariableOp"transformer_model_3/dense_1/kernel*
_output_shapes
:	*
dtype0

transformer_model_3/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name transformer_model_3/dense/bias

2transformer_model_3/dense/bias/Read/ReadVariableOpReadVariableOptransformer_model_3/dense/bias*
_output_shapes	
:*
dtype0

 transformer_model_3/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" transformer_model_3/dense/kernel

4transformer_model_3/dense/kernel/Read/ReadVariableOpReadVariableOp transformer_model_3/dense/kernel* 
_output_shapes
:
*
dtype0
Ý
Btransformer_model_3/transformer_encoder/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBtransformer_model_3/transformer_encoder/layer_normalization_1/beta
Ö
Vtransformer_model_3/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOpBtransformer_model_3/transformer_encoder/layer_normalization_1/beta*
_output_shapes	
:*
dtype0
ß
Ctransformer_model_3/transformer_encoder/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECtransformer_model_3/transformer_encoder/layer_normalization_1/gamma
Ø
Wtransformer_model_3/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOpCtransformer_model_3/transformer_encoder/layer_normalization_1/gamma*
_output_shapes	
:*
dtype0
Ã
5transformer_model_3/transformer_encoder/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75transformer_model_3/transformer_encoder/conv1d_1/bias
¼
Itransformer_model_3/transformer_encoder/conv1d_1/bias/Read/ReadVariableOpReadVariableOp5transformer_model_3/transformer_encoder/conv1d_1/bias*
_output_shapes	
:*
dtype0
Ð
7transformer_model_3/transformer_encoder/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97transformer_model_3/transformer_encoder/conv1d_1/kernel
É
Ktransformer_model_3/transformer_encoder/conv1d_1/kernel/Read/ReadVariableOpReadVariableOp7transformer_model_3/transformer_encoder/conv1d_1/kernel*$
_output_shapes
:*
dtype0
¿
3transformer_model_3/transformer_encoder/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53transformer_model_3/transformer_encoder/conv1d/bias
¸
Gtransformer_model_3/transformer_encoder/conv1d/bias/Read/ReadVariableOpReadVariableOp3transformer_model_3/transformer_encoder/conv1d/bias*
_output_shapes	
:*
dtype0
Ì
5transformer_model_3/transformer_encoder/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75transformer_model_3/transformer_encoder/conv1d/kernel
Å
Itransformer_model_3/transformer_encoder/conv1d/kernel/Read/ReadVariableOpReadVariableOp5transformer_model_3/transformer_encoder/conv1d/kernel*$
_output_shapes
:*
dtype0
Ù
@transformer_model_3/transformer_encoder/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@transformer_model_3/transformer_encoder/layer_normalization/beta
Ò
Ttransformer_model_3/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpReadVariableOp@transformer_model_3/transformer_encoder/layer_normalization/beta*
_output_shapes	
:*
dtype0
Û
Atransformer_model_3/transformer_encoder/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAtransformer_model_3/transformer_encoder/layer_normalization/gamma
Ô
Utransformer_model_3/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpReadVariableOpAtransformer_model_3/transformer_encoder/layer_normalization/gamma*
_output_shapes	
:*
dtype0
ñ
Ltransformer_model_3/transformer_encoder/multi_head_attention/projection_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLtransformer_model_3/transformer_encoder/multi_head_attention/projection_bias
ê
`transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/Read/ReadVariableOpReadVariableOpLtransformer_model_3/transformer_encoder/multi_head_attention/projection_bias*
_output_shapes	
:*
dtype0
þ
Ntransformer_model_3/transformer_encoder/multi_head_attention/projection_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*_
shared_namePNtransformer_model_3/transformer_encoder/multi_head_attention/projection_kernel
÷
btransformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/Read/ReadVariableOpReadVariableOpNtransformer_model_3/transformer_encoder/multi_head_attention/projection_kernel*$
_output_shapes
:*
dtype0
ô
Itransformer_model_3/transformer_encoder/multi_head_attention/value_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKItransformer_model_3/transformer_encoder/multi_head_attention/value_kernel
í
]transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/Read/ReadVariableOpReadVariableOpItransformer_model_3/transformer_encoder/multi_head_attention/value_kernel*$
_output_shapes
:*
dtype0
ð
Gtransformer_model_3/transformer_encoder/multi_head_attention/key_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGtransformer_model_3/transformer_encoder/multi_head_attention/key_kernel
é
[transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/Read/ReadVariableOpReadVariableOpGtransformer_model_3/transformer_encoder/multi_head_attention/key_kernel*$
_output_shapes
:*
dtype0
ô
Itransformer_model_3/transformer_encoder/multi_head_attention/query_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKItransformer_model_3/transformer_encoder/multi_head_attention/query_kernel
í
]transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/Read/ReadVariableOpReadVariableOpItransformer_model_3/transformer_encoder/multi_head_attention/query_kernel*$
_output_shapes
:*
dtype0
Ã
5transformer_model_3/positional_embedding/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75transformer_model_3/positional_embedding/dense_2/bias
¼
Itransformer_model_3/positional_embedding/dense_2/bias/Read/ReadVariableOpReadVariableOp5transformer_model_3/positional_embedding/dense_2/bias*
_output_shapes	
:*
dtype0
Ë
7transformer_model_3/positional_embedding/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*H
shared_name97transformer_model_3/positional_embedding/dense_2/kernel
Ä
Ktransformer_model_3/positional_embedding/dense_2/kernel/Read/ReadVariableOpReadVariableOp7transformer_model_3/positional_embedding/dense_2/kernel*
_output_shapes
:	*
dtype0
â
ConstConst* 
_output_shapes
:
*
dtype0*¡
valueB
"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?¤jW?^MM? C?®8?Îz.?~$?:Í?ut?º}?ûÝÿ>äï>G(à>ðÑ>)ÍÃ>`×¶>®¨ª>>9>£>v>µ>'p>Ù#`>aÎP>(~B>:#5>R®(>ä>!=>ó%>~ý=]ùë=¨Û=vuÌ=dN¾=¿ ±=¡Û¤=7o=¼Ì=`æ=^w=°6f=Ù>V=bG=Ö9=û¬,=± =Æ=ð)=\=ñ<¥Mà<ß»Ð<¦>Â<ÐÂ´<6¨<&<B«<r<¦J|<ÙÆj<IzZ<^OK<þ1=<t0<WÖ#<uv<¹à<<¹õ;Ì©ä;²ÉÔ;¿Æ;\D¸;Py«;£;};].;h;êQo;^´^; >O;³Ú@; w3;G';i;Ó;i;çxú: é:læØ:W×É:èÓ»:É®:ó¦¢:\:ÙÙ:m: ñs:àc:&?S:D:î6:C;*:¤i:j:.	:¬Oÿ9âí9%Ý9½Í9×t¿9û)²9_Ë¥9H9r9©9¨x9dg9äSW9Á`H9Sw:9(-9y!9#C9qÔ9+9ã,ò8\á8@Q
?í?)Ý%?ðG1?°T;?,(D?ªäK?¶©R?X?¨½]?Õ=b?`)f?Æi?dl?©o?I\q?jNs?Ôþt?vv?»w?ðÕx?ÅÊy?	z?	W{?ö{?¸|?|ø|?@`}?(º}?~?K~?÷~?¸~?rä~?n
?S+?ÑG?`?Þu?`?h?J¦?O²?¹¼?½Å?Í?OÔ?*Ú?<ß? ã?nç?¹ê?í?ð?/ò?	ô?£õ?÷?;ø?Eù?,ú?ôú?¡û?7ü?¹ü?*ý?ý?ßý?(þ?hþ?þ?Îþ?÷þ?ÿ?9ÿ?Tÿ?kÿ?ÿ?ÿ?ÿ?¬ÿ?·ÿ?Áÿ?Êÿ?Ñÿ?×ÿ?Ýÿ?áÿ?åÿ?éÿ?ìÿ?ïÿ?ñÿ?óÿ?õÿ?öÿ?øÿ?ùÿ?úÿ?ûÿ?ûÿ?üÿ?üÿ?ýÿ?ýÿ?þÿ?þÿ?þÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?·Çh?óHu?¯|?:É?ÙZ?|?v?êco?
òf?1 ]?½S?kI?5??ûê4?ùÈ*?Qæ ?T?m?VM?fËù>¶Óé>g´Ú>BkÌ>Ðó¾>öG²>i`¦>5>e½>ð>{>gj>©cZ>ùoK>{=>Öv0>¬R$>Ó >Ãs>ª>ëêö=Ùå=ÅóÕ=%Ç=\¹=®¬=â =@q=_=Èp=×íp=8`=ªP=«0B=·4=o-(=Õ=\¥=²=ýBü<­Àê<PuÚ<\KË<Ã.½<Ù°<>Ô£<Ät<\ß<<T·u<_¨d<ÈT<ÒF<C8<¶x+<'<)}<.
<' <Qï;	´Þ;Ý=Ï;|ÚÀ;Ôv³;$§;æh;¼;V;Éxz;i;YæX;G×I;ÛÓ;;É.;ë¦";\;ÔÙ;i;ñó:Úã:!?Ó:Ä:î¶:@;ª:¢i:j:.:ªO:ám:$]:½M:Öt?:ú)2:_Ë%:H:r:©:¨ø9dç9äS×9Á`È9Swº9(­9y¡9#C9qÔ9+9ã,r9\a93Õ¾¾SI$¾Hn'½QM=Kl2>Æ>|kµ>²èÜ>" ?uâ?¾Ü?9*?ç5?´>?ÅG?ì{N?íT?<Z?w_?i¾c?(xg? ¶j?m?¥úo?Xr?Iôs?Çu?õòv?í'x?â3y?9z?¥åz?<{?+|?¯®|?O }?Â}?Ø}?ñ!~?óa~?f~?mÉ~?ó~??C6?JQ?³h?ù|??¼?çª?N¶?/À?¼È?$Ð?Ö?Ü?ìà?å?±è?Ñë?î?Ýð?åò?¦ô?,ö?}÷?¡ø?ù?yú?7û?Ûû?iü?äü?Oý?«ý?ûý?Aþ?}þ?°þ?Ýþ?ÿ?&ÿ?Cÿ?]ÿ?sÿ?ÿ?ÿ?¤ÿ?°ÿ?»ÿ?Äÿ?Ìÿ?Óÿ?Ùÿ?Þÿ?ãÿ?çÿ?êÿ?íÿ?ðÿ?òÿ?ôÿ?õÿ?÷ÿ?øÿ?ùÿ?úÿ?ûÿ?üÿ?üÿ?ýÿ?ýÿ?ýÿ?þÿ?þÿ?þÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?  ?  ?Ã>¯>&n?j)?Þ<G?kÑ]?4ým?Ýx?×E~?ùÿ?¬t~?!Nz?c!t?Fnl?z c?OZ?Â	P?ÄE?p;?ñ11?¸$'?½]?$ì?oÚ
?m/?÷Ýó>\5ä> dÕ>§hÇ>Q<º>Ù­>)7¢>mN> >_>&u>âmd>¤ÏT>;F>ê8>Øî+> >>'È
>,3>²ð=êß=©lÐ=ÁÿÁ=ÿ´=/¨=6j=={=u-|=v°j=MiZ=ªBK=¬(==Ë0=ÅÑ#=s=	ß=b= ¹õ<íªä<£ËÔ<EÆ<IG¸<|«<ú<ö<Ç1<È<Xo<Ãº^<IDO<à@<|3<'<n<£<è<]ú;é;àíØ;SÞÉ;qÚ»;¨Ï®;ª¬¢;ba;ÕÞ;;Lús;÷	c;°FS;D;*õ6;]A*;So;ao;ð2	;ÛXÿ:oí:Ý:þÄÍ:½{¿:g0²:YÑ¥:.N: :{:±x:êlg:ª[W:ügH:~::l-:ë~!:H:~Ù:Þ#:¡5ò9µdá9ª¾Ñ9Á.Ã9ª¡µ9m©9&p}¿}p¿Á[¿9Æ?¿Í¿ ¿oÿ¾a¦¼¾}u¾¨í½´©k:à=6ÉV>c>ÓTÄ>2Hê>R?W/?Ó"?â\.?³Ã8?¢çA?¦ëI?ðP?QW?)m\?a?	*e?´h?7Ék?Èvn?vÊp?ÍÏr?ït?¶v?âhw?4x?y?iz?K({?Î{?]|?Ú|?åE}?S£}?Cô}?b:~?w~?¼«~?OÙ~?È ?ù"?@?;Z?qp?­?V?Ã¢?B¯?º?sÃ?Ë?Ò?­Ø?òÝ?â?wæ?ãé?Úì?kï?¤ñ?ó?;õ?­ö?í÷?ù?òù?Áú?uû?ü?ü?ý?rý?Êý?þ?Wþ?þ?Âþ?ìþ?ÿ?1ÿ?Mÿ?eÿ?zÿ?ÿ?ÿ?©ÿ?µÿ?¿ÿ?Çÿ?Ïÿ?Öÿ?Ûÿ?àÿ?äÿ?èÿ?ëÿ?îÿ?ðÿ?óÿ?ôÿ?öÿ?÷ÿ?øÿ?ùÿ?úÿ?ûÿ?üÿ?üÿ?ýÿ?ýÿ?þÿ?þÿ?þÿ?þÿ?ÿÿ?ÿÿ?ÿÿ?ÿÿ?Ï½A¿jq¿(¢¾uJ§½ï>±¯>³?'¦)?íIG?Û]?Ön?áx?¥G~?õÿ?0s~?~Kz?Þt?jl?×c?_Z?¦P?p¿E?^k;?×,1?¼'?éX?~ç?ûÕ
?-+?åÕó>¶-ä>f]Õ>×aÇ>è5º> Ó­>1¢> I>*>»>Òu>Æed>ÈT>õ3F>R8>²è+>Y >C>,Ã
>.>ð=âß=)eÐ=ÅøÁ=´=!	¨=d=È=±v=]$|=ÿ§j=laZ=U;K=Ø!==q0=ÛË#=n=êÙ==A°õ<¬¢ä<ôÃÔ<ÿÅ<¢@¸<Nv«<7<{<Ê,<#<àOo<¸²^<Í<O<¡Ù@<$v3< '<th<`<<Qxú;§é;æØ;×É;©Ó»;XÉ®;Ê¦¢;ê[;¿Ù;X;}ñs;Äc;?S;yD;î6;7;*;i;j;ü-	;¢Oÿ:Úí:Ý:½Í:Ót¿:÷)²:\Ë¥:H:p:§:¨x:dg:ãSW:À`H:Rw::(-:y!:#C:qÔ:+:â,ò9\á90U'¿l	V¿?Òr¿þ$¿;l}¿§tp¿S	[¿k·?¿¯ ¿åwÿ¾å¼¾ <u¾¼í½*Û:vá=\úV>±.>íhÄ>4Zê>Y!?x6?!"?qb.?È8?íëA?iïI?WóP?1W?«o\?Äa?ð+e?D¶h?§Êk?xn?Ëp?¿Ðr?Àt?lv?iw?½x?	y?iz?¥({?NÎ{?á]|?KÚ|?F}?~£}?iô}?:~?;w~?Õ«~?dÙ~?Û ?	#?£@?GZ?{p?¶?^?Ê¢?H¯?º?wÃ?Ë?Ò?°Ø?õÝ?â?xæ?äé?Ûì?lï?¥ñ?ó?<õ?­ö?í÷?ù?òù?Âú?vû?ü?ü?ý?rý?Êý?þ?Xþ?þ?Âþ?ìþ?ÿ?1ÿ?Mÿ?eÿ?zÿ?ÿ?ÿ?©ÿ?µÿ?¿ÿ?Çÿ?Ïÿ?Öÿ?Ûÿ?àÿ?äÿ?èÿ?ëÿ?îÿ?ñÿ?óÿ?ôÿ?öÿ?÷ÿ?øÿ?ùÿ?úÿ?ûÿ?üÿ?üÿ?ýÿ?ýÿ?þÿ?þÿ?þÿ?þÿ?|u¿ê¿[~m¿ÉF¿d5¿Æc®¾¤I×½Leõ=ÖÚ¥> U ?=&?[D?à[?o¦l?·w?Iå}?ü?K¼~?èÐz?Ñt?ø?m?íd?l
[?øQ?«ÊF?Kv<?142?	!(?TR?º×?'¼?¿?Dwõ>.¹å>$ÓÖ>FÂÈ>»>¸
¯>	V£>T[>î>öp>ßv>v	f>ùOV>À¡G>î9>þ&-><!>å>øÄ>¼>Bò=èá=déÑ=LbÃ=!Ûµ=B©=`=n=s=$û}=3^l=Kù[=õ¶L=,>=IK1=èý$=Ø=ÿâ=Qø=n{÷<þMæ<¡QÖ<3qÇ<¹<Ï¶¬<y¹ <(</<<<q<S`<6ÀP<%BB<¡Å4<É8(<û<»¬<¢<Lü;dÈê;{Ú;^PË;Ì2½;°;ÞÖ£;áv;á;a;¹u;'ªd;üÉT;úF;D8;vy+;Â;¥};q.
;x ;Rï:s´Þ:1>Ï:ÀÚÀ:w³:P§:
i:Ù:m:ïxz:&i:qæX:[×I:ëÓ;:É.:ö¦":\:ÛÙ:,<>¯s½¢!¿¾R!¿ü#R¿_±p¿å~¿Ö'~¿@2r¿®]¿]®B¿ð#¿D¿pEÃ¾¼/¾ë¾Ôr)¼MË=M>yª>VÀ>´æ>ã?¬Ä?ñL!?ÖA-?«Ê7?A?,I?zHP?ïV?oí[?@©`?Éd?>`h?Ùk?7n?p?µr?/gt?|òu?uIw?órx?ðty?Tz?{?¾{?GP|?Î|?â;}?¥}?¿ì}?ß3~?{q~?Ú¦~?Õ~?ý~?Ì?Õ=?ÙW?an?ä?Ê?l¡?®?¹?Â?ÏÊ?ðÑ?Ø?uÝ?â?æ?é?ì?.ï?oñ?có?õ?ö?Ï÷?èø?Ûù?®ú?eû?ü?ü?ý?iý?Âý?þ?Qþ?þ?½þ?èþ?ÿ?.ÿ?Jÿ?cÿ?xÿ?ÿ?ÿ?¨ÿ?³ÿ?¾ÿ?Çÿ?Îÿ?Õÿ?Ûÿ?àÿ?äÿ?èÿ?ëÿ?îÿ?ðÿ?òÿ?ôÿ?öÿ?÷ÿ?øÿ?ùÿ?úÿ?ûÿ?üÿ?üÿ?ýÿ?ýÿ?þÿ?¾ß$¿öªb¿~¿¢6z¿x]¿`/¿kTî¾tåk¾®©ë:9A_>Ò>Wó?ÄR5?àPP?Bud?Mxr?ù'{?ÿT?´Æ?Ô3}?é>x?#uq?ÅNi?0`?mV?4IL?½ùA?Í©7?}z-?ó#?ÑÛ?i?Ç?"7þ>zî>P¬Þ>'*Ð>{Â>[µ>ý}©>!>z>X>Ù*>ván>w^>¢WO>o A>üÜ3>~'>Bõ>Ä4>/>þ²û=ÂMê=ÁÚ=Ë=õ¼=(ß¯=7°£=xX=>É=Óô=Ñu=d=A¹T=u÷E=Y;8=ár+=9=¶z=¿,
=¹ =úQï<~µÞ<
@Ï<,ÝÀ<Ýy³<e§<Fl<'¢<¾<}z<i<©ìX<XÝI<§Ù;<Ï.<'¬"<ø`<Þ<Ï<Þùó;	ã;iFÓ;YÄ;ûô¶;7Aª;5o;Io;Ü2;»X;Um;];íÄM;°{?;\02;QÑ%;'N;;v;±ø:älç:¥[×:øgÈ:~º:i­:é~¡:H:}Ù:Ý#: 5r:³da:©¾Q:À.C:©¡5:l):¸Íu?×C?çøí>/ú=}X¾}f ¿}:¿ëb¿Sy¿åÿ¿t×y¿`yi¿TQ¿·4¿øË¿íç¾o?¤¾AF¾¶Ù½Ý<+=ø>#z>Õª>¿Ò>4/÷>oØ?uH?³'?V2?&B<?{øD?L?2IS?OY?7^?©§b?f?îâi?Ðl?8[o?âq?|s?u&u?sv?lÙw?Îïx?3áy?z²z?ãg{?|?a|?s}?Ái}?cÂ}?-~?²Q~?Q~?:½~?vè~?è?V.?mJ?Áb?Ów??à?§?i³?­½?Æ?CÎ?íÔ?³Ú?³ß?ä?Çç?ë?Öí?Eð?aò?4ô?Éõ?'÷?Wø?^ù?Aú?û?±û?Eü?Åü?4ý?ý?çý?/þ?mþ?£þ?Òþ?úþ?ÿ?<ÿ?Vÿ?mÿ?ÿ?ÿ?¡ÿ?­ÿ?¸ÿ?Âÿ?Êÿ?Ñÿ?Øÿ?Ýÿ?âÿ?æÿ?éÿ?ìÿ?ïÿ?ñÿ?óÿ?õÿ?öÿ?øÿ?ùÿ?úÿ?ûÿ?ûÿ?üÿ?ýÿ?F0(?õCj>¦æ`¾Z¿øû[¿44|¿Ts|¿èËb¿n¶6¿Àÿ¾¨,¾òù¼hA>lÜÄ>\q?4ä0?oèL?éûa?öÑp?Î6z?§û~?¶é?\º}?#y?Dr?ìj?a?uØW?h¾M?=rC?% 9?|ê.?së$?6?´Ù?Þ?LK ?/Fð><Îà>¬-Ò>ß`Ä>Vb·>5+«>°³>`ó>á>.u>ûJq>PÓ`>4rQ>C>Û±5>U3)>æ>¶°>¨>ºFþ=S´ì=*VÜ=£Í=hå¾=\­±=^¥=é=2>=ÿO=$#x=¶íf=3éV= H=h :=R6-=`1!=¼=§=dè=TÈñ< á<õaÑ<6ÙÂ<¦Rµ<^¼¨<·<-<Rú<k}<®k<'([<*ñK<È=<0<ºX$<Ëï<¢Q<-p<§|ö;Ç_å;sÕ;V¡Æ;×¸;Ç¬;¡ ;¹ó;U;¾ü;ap;e_;ãO;.tA;Ô4;1';²ä;ì;ÿ;>@û: Îé:Ù:úwÊ:di¼:¥T¯:f(£:Ô:òI:¾z:Ä³t:¶c:EçS:þ0E:½ÿ@?ñ5y?Ó¿y?büL?Wî?°/>Õí)¾6{í¾`O3¿sø]¿¼êv¿}á¿!h{¿Rl¿ `U¿!9¿»t¿<ð¾t¯­¾WzX¾\^¶½¸¤Õ<ã>[l>¹ø£>Ì9Í>÷>ò>¤¥	?aU?V%?ZÑ0?mì:?³ÌC?K?©cR?âVX?M]?Vb?å f?oi?Åkl?o?,Eq?Z:s?iít?ôfv?}®w?Êx?êÀy?}z?¡O{?ð{?({|?ªó|?\}?¶}?é~?ÍH~?~?¶~?®â~?æ?*?¬F?_?u?¢?Ã?»¥?Ô±?M¼?`Å?;Í?	Ô?îÙ?ß?sã?Gç?ê?ví?òï?ò?öó?õ?ùö?.ø?;ù?#ú?ìú?û?1ü?´ü?%ý?ý?Üý?%þ?eþ?þ?Ìþ?õþ?ÿ?8ÿ?Sÿ?jÿ?~ÿ?ÿ?ÿ?¬ÿ?·ÿ?Áÿ?Éÿ?Ñÿ?×ÿ?Üÿ?áÿ?åÿ?éÿ?ìÿ?ïÿ?ñÿ?óÿ?õÿ?öÿ?øÿ?ùÿ?úÿ?ûÿ?ûÿ?F}?ô×j?|Ï?W»&>z¾%¿EÁb¿~¿Ó-z¿ e]¿`F/¿î¾
lk¾$Û;¨ª_>â/Ò>}?!b5?¦\P?Ã}d?í}r?!+{?V?Æ?ã1}?ë;x?Zqq?hJi?Å+`?|hV?DL?ôA?¦¤7?mu-?#?×?Õ?f?Ì.þ>ýí>Ò¤Þ>#Ð>itÂ>µ>x©>>ðt>{>R&>ÿØn>^>@PO>A>Ö3>x'>«ï>/>À*>ó©û=VEê=éÚ=@ûÊ=4î¼=ÒØ¯=Qª£=úR=!Ä=ð=öu=ßd=±T=PðE=³48=±l+=w=Zu=Â'
= =VIï<t­Þ<8Ï<6ÖÀ<bs³<]þ¦<ªf<î<â<rvz<%i<ÓäX<ÖI<ßÒ;<µÈ.<G¦"<[<iÙ<<ñó;kã;Ç>Ó;?Ä;`î¶;;ª;|i;öi;è-;O;Ám;
];½M;Æt?;í)2;TË%;H;k;£;
¨ø:dç:ÞS×:¼`È:Owº:%­:y¡:!C:pÔ:*:à,r:\a:öý¾ÏË>X¤L?n|?3¾u?ÝµC?Õ£í>J3ù= Y¾÷ ¿E:¿¢b¿$y¿Öÿ¿Ñy¿oi¿æQ¿§4¿}»¿Häæ¾5¤¾F¾`½Ï,=¢,>Nz>R1ª>jÒÒ>z@÷>à?FO?¹'?_[2?ÐF<?üD?§L?TLS?"Y?9^?½©b?Rf?äi?sÑl?h\o?êq?ú|s?<'u? v?Úw?Pðx?£áy?Ü²z?8h{?h|? |?ª}?ñi}?Â}?Q~?ÑQ~?l~?R½~?è~?ù?f.?zJ?Ìb?Ýw??ç?§?o³?²½?Æ?FÎ?ðÔ?¶Ú?µß?	ä?Éç?ë?×í?Fð?bò?5ô?Êõ?(÷?Xø?^ù?Bú?û?²û?Eü?Åü?4ý?ý?çý?/þ?mþ?£þ?Òþ?ûþ?ÿ?<ÿ?Vÿ?mÿ?ÿ?ÿ?¡ÿ?­ÿ?¸ÿ?Âÿ?Êÿ?Òÿ?Øÿ?Ýÿ?âÿ?æÿ?éÿ?ìÿ?ïÿ?ñÿ?óÿ?õÿ?öÿ?øÿ?ùÿ?úÿ?2Ó>^^?á?CS?µüå>î1»Ù¾f?¿Fp¿Ãÿ¿6r¿pAN¿­¿¦½Â¾5ï¾&5ª=ê>`ò>H` ?Á@?¡mX?>j?7v?D)}?/ã?­&?^£{?ôu?"n?f?j¯\?ÌÁR?H?4>?©ì3?æÏ)?fô?¹j?>?w?4ø>òQè>GÙ>Ë>±¯½>A±>]B¥>à(>.Â>l>ZÓy>.Ìh>õãX>Ì	J>d-<>?/>ÿ/#>Üñ>:w>\³>|4õ=#Aä=iwÔ=YÃÅ=?¸=R«=øs=g==#=u@o=t¨^=6O=rÖ@=Pu3='=¹j=¡=ó=	ú<ïé<ïØ<µàÉ<FÝ»<ÈÒ®<ù¯¢<Èd<@â<x<ò t<kc<êLS<¡D<Üú6<ÇF*<tt<:t<7	<uaÿ;¦í;°&Ý;ÌÍ;d¿; 6²;+×¥;S;³;7¤;ã¹x;(ug;YcW;&oH;»:;¤-;¶!;õM;Þ;(;X>ò:Òlá:8ÆÑ:É5Ã:6¨µ:©:ýN:c:>9:!}:Õ?i¿¾ëþ¾¬Ùv=4Ø?¸d?Âÿ?¢Àg?í*?î®>:¿0»qÁ¥¾É£¿:K¿gÂl¿P}¿F¿¢æt¿~a¿ÉG¿ÞP)¿º¿Î¾Ù3¾ÿ¾:æò¼Æ¥¦=
9<>%>Fr¹>Óà>þ¾?Q?Õ!?ýW+?6?r??OàG?-&O?IU?[?Kè_?!d?ýÍg?§ k?vÈm?3p?MLr?Ét?©³u?ôw?®Cx?óKy?1z?»÷z?é£{?$9|?uº|?*}?}?µß}?(~?²g~?`~?¼Í~?Âö~?J?9?·S?Íj?Ë~???¬?U·?Á?É?ÎÐ?"×?Ü?Zá?vå?é?ì?Äî?ñ?ó?Ïô?Oö?÷?»ø?µù?ú?Hû?êû?vü?ïü?Yý?´ý?þ?Gþ?þ?µþ?áþ?ÿ?)ÿ?Fÿ?_ÿ?tÿ?ÿ?ÿ?¥ÿ?²ÿ?¼ÿ?Åÿ?Íÿ?Ôÿ?Úÿ?ßÿ?ãÿ?çÿ?ëÿ?íÿ?ðÿ?òÿ?ôÿ?öÿ?÷ÿ?øÿ?øD¿Aó=yP1?3«z?³p?^ö#?HV>± s¾lé¿Ç^¿×|¿¶Ó{¿m8a¿÷4¿^ú¾0ó¾p©¼vLJ>¦àÈ>ö ?8@2?´ôM?N¿b?òTq?kz?Ð?Çà?ç}?ÒÓx?Õ2r?;*j? a?¤jW?^MM? C?®8?Îz.?~$?:Í?ut?º}?ûÝÿ>äï>G(à>ðÑ>)ÍÃ>`×¶>®¨ª>>9>£>v>µ>'p>Ù#`>aÎP>(~B>:#5>R®(>ä>!=>ó%>~ý=]ùë=¨Û=vuÌ=dN¾=¿ ±=¡Û¤=7o=¼Ì=`æ=^w=°6f=Ù>V=bG=Ö9=û¬,=± =Æ=ð)=\=ñ<¥Mà<ß»Ð<¦>Â<ÐÂ´<6¨<&<B«<r<¦J|<ÙÆj<IzZ<^OK<þ1=<t0<WÖ#<uv<¹à<<¹õ;Ì©ä;²ÉÔ;¿Æ;\D¸;Py«;£;};].;h;êQo;^´^; >O;³Ú@; w3;G';i;Ó;i;çxú: é:læØ:W×É:èÓ»:É®:ó¦¢:\:ÙÙ:dÍV¿0~¿f¦8¿=äO¾9ü±>D?Wz?(¦x?FJ?«þ>Üe >8#8¾>hó¾5¿ng_¿ýw¿ûñ¿`ôz¿_yk¿@T¿¿7¿ñ¿U«í¾-Òª¾
ñR¾Fß«½bØü<¥>1²p>n×¥>èÎ>cÀó>@Q
?í?)Ý%?ðG1?°T;?,(D?ªäK?¶©R?X?¨½]?Õ=b?`)f?Æi?dl?©o?I\q?jNs?Ôþt?vv?»w?ðÕx?ÅÊy?	z?	W{?ö{?¸|?|ø|?@`}?(º}?~?K~?÷~?¸~?rä~?n
?S+?ÑG?`?Þu?`?h?J¦?O²?¹¼?½Å?Í?OÔ?*Ú?<ß? ã?nç?¹ê?í?ð?/ò?	ô?£õ?÷?;ø?Eù?,ú?ôú?¡û?7ü?¹ü?*ý?ý?ßý?(þ?hþ?þ?Îþ?÷þ?ÿ?9ÿ?Tÿ?kÿ?ÿ?ÿ?ÿ?¬ÿ?·ÿ?Áÿ?Êÿ?Ñÿ?×ÿ?Ýÿ?áÿ?åÿ?éÿ?ìÿ?ïÿ?ñÿ?óÿ?õÿ?öÿ?\ÿ¿D¯9¿8&Î½Y?+Ml?Pö{?B?ÈP¶>ü8Ç½çº ¿0¯L¿bv¿öM¿Ï5l¿ID¿nË¿!A©¾Ã½í>>ª>¹?5®'?UÀE?g·\?;m?´x?ä~?ÿ?m~?z?t?~æl?*&d?ÝZ?qP?qZF?&<?Å1?ß´'?ué?³r?X[?cª?¯Çô>Íå>ä5Ö>û-È>
öº>®>îÚ¢>ïç>Þ¥>Û>ö!v>ÓXe>«U>ËG>õ^9>¡,>5¿ >u«>vX>¡¹>6ñ=rÓà=ûEÑ= ÊÂ=oMµ=¡¾¨==+=!	=û4}=À¥k=M[=*L=sî==ÞÀ0=}$=ó=ks=z='ºö< å<=ªÕ<mÕÆ<¹<ç/¬<î; <S<\Â<T <îRp<Ë£_<$P<eªA<j84<_µ'<±<î:<¼%<xû;úê;ÚÐÙ;±Ê;ÿ¼;¯;àV£;Æÿ;:r;= ;ùt;÷c;À#T;IiE;´7;ó*;;¥	;~Â	;2 ;î:xÞ:KÎ:DÀ:Øê²:Ù~¦:¢ï:r;;0¿%³~¿ÐÓX¿ÿóÄ¾®'5>'?Y8o?0É~?G]?Á?+º>gÛ½õcÅ¾"$¿·ÌS¿q¿ôÕ~¿«Ü}¿Ñyq¿z\¿ÉpA¿
"¿f±¿oÀ¾Ù|¾Ýû½êË´»nsÔ=.8Q>>ÇÂ>À=è>j0?`?ßÖ!?»-?»58?kA?u~I?P?à¾V?[$\? Ù`?Æòd?h?fk?pRn?äªp?d´r?#yt?v?ùVw?¬~x?y?o]z?,{?;Å{?V|?zÓ|?0@}?a}?ûï}?¬6~?ès~?ó¨~?æÖ~?²þ~?*!???ßX?Do?¨?t? ¢?®?¹?ôÂ?"Ë?8Ò?[Ø?«Ý?Eâ?Aæ?´é?±ì?Hï?ñ?vó?%õ?ö?Ü÷?óø?åù?¶ú?lû?	ü?ü?ý?mý?Åý?þ?Tþ?þ?¿þ?êþ?ÿ?0ÿ?Lÿ?dÿ?yÿ?ÿ?ÿ?¨ÿ?´ÿ?¾ÿ?Çÿ?Ïÿ?Õÿ?Ûÿ?àÿ?äÿ?èÿ?ëÿ?îÿ?ðÿ?òÿ?ôÿ?Ø\	¿í@|¿¨´R¿,´x¾ÂÓ>à)^??MïR?3å>©k»UâÙ¾m?¿>¨p¿¡ÿ¿x)r¿+N¿&¿AÂ¾ôv¾«=@R>¡ò>
r ?Æ@?*xX?{Ej?µ;v? +}?ã?%?¡{?¼ðu?n?f?ª\?¸¼R?ñH?ä.>?ç3?áÊ)?ï?f?:?7s?e,ø>/Jè>A@Ù>"Ë>,©½> ±><¥>|#>!½>² >Êy>ìÃh>?ÜX>J>¬&<>Ö8/>'*#>jì>'r>¡®>¬+õ=î8ä=ÄoÔ=:¼Å=¸=iL«=9n=»a===Ô7o=l ^=/O=}Ï@=Ön3=û&=e=b==þwú<é<ÊçØ<kÙÉ<~Ö»<xÌ®<ª¢<P_<*Ý<¼<"øs<8c<IES<qD<@ô6<¡@*<¼n<çn<2	<<Xÿ;ïí;´Ý;«ÄÍ;z{¿;10²;.Ñ¥;N;;d;è°x;Ílg;[W;égH;ÿ}:;_-;á~!;H;wÙ;Ù#;5ò:®dá:¤¾Ñ:¼.Ã:¦¡µ:j©:ÐX?r.>d¿³Ux¿i¿ieþ¾FX{=w?Ôd?ÿ?ª©g?SÜ)?&®>¦H\»¦¾'Â¿UOK¿ÐÎl¿âT}¿ë¿÷Þt¿æsa¿{G¿÷@)¿ñ©¿mÎ¾ü¾Ô¾ ñ¼§=ik<>>õ¹>bà>FÇ?éX?Y(?½]+? 6?ä??5äG?)O?EU?±[?ê_? #d?µÏg?%k?ÃÉm?'4p?HMr?£t?f´u?w?<Dx?nLy?1z?øz?:¤{?i9|?±º|?·*}?Æ}?Üß}?µ(~?Ðg~?y~?ÒÍ~?Õö~?[?9?ÄS?×j?Ô~?#? ?¬?Z·?Á?É?ÒÐ?%×?Ü?\á?xå?é?ì?Åî?ñ?ó?Ðô?Oö?÷?¼ø?µù?ú?Hû?êû?vü?ðü?Yý?´ý?þ?Gþ?þ?µþ?âþ?ÿ?)ÿ?Fÿ?_ÿ?uÿ?ÿ?ÿ?¥ÿ?²ÿ?¼ÿ?Åÿ?Íÿ?Ôÿ?Úÿ?ßÿ?ãÿ?çÿ?ëÿ?íÿ?ðÿ?òÿ?" ×>dç¾(Dw¿5^¿+í¢¾ÿ°>U?ãÿ?[?OEþ>B=¿©Ã¾@Ø7¿KÞl¿&Ø¿Zöt¿LS¿®Á!¿Ç Ð¾^r.¾ç»p=o÷>~Ëè>l?á<?­V?«h?'u?£|?	Å?la?$|?À«v?~o?Jg?ÛÀ]?6ßS?F«I?X??5?êê*?L!?ïs?ø<?j?ëú>^ê>4æÚ>@Ì> ¿>q²>z¦>¶Y>´ß>>Ç{>%j>Z>ç K>B©=>l¡0>az$>Ø%>C>Ð¾>Ñ&÷=Sæ=¼'Ö=ôUÇ=#¹=£¯¬=ò¸ ==97=L=(q=¸n`=YÝP=ù_B=ã4=iV(=÷§=ÚÈ=¹ª=wü<ãùê<ªÚ<è|Ë<ß\½<Â7°<-ü£<í<ñ<0&<9óu<àd<küT<3F<p8<¢+<¸<[¡<»O
<µ <Ùï;SêÞ;bpÏ;	Á;¢³;Ú)§;É;ýÁ;%µ;Øµz;ÙMi;9Y;{J;¥<;ô.;Î";ë;*ü;]2;-ô:19ã: rÓ:sÄÄ:2·:oNh?l]d?><þ¾T±r¿{7p¿ÿ¿
Kõºpe?3^?j¶?¸l?.&2?Í5Â>UÑ=¸¾Ù¿¹kF¿"ìi¿B|¿·¿Òv¿
d¿§J¿èÌ,¿Õd¿ôÕ¾r¾ ô%¾&¶-½fb=11>õ>eÞ´>jÜ>æÒÿ>+°?4°?¯*?Gü4?>?G?dN?ÙT?n|Z?g_?ã°c?clg?Ã«j?«~m?êòo?¡r?uîs?¸u?îv?$x?0y?\z?*ãz?{?±)|?­|?é}?}? ×}?!~?*a~?·~?ÕÈ~?ò~??á5?õP?ih?¹|?P??¾ª?+¶?À?¡È?Ð?zÖ?Ü?Üà?	å?¦è?Çë?}î?Öð?Þò?¡ô?'ö?y÷?ø?ù?vú?4û?Ùû?gü?ãü?Ný?ªý?úý?@þ?|þ?°þ?Ýþ?ÿ?&ÿ?Cÿ?\ÿ?rÿ?ÿ?ÿ?¤ÿ?°ÿ?»ÿ?Äÿ?Ìÿ?Óÿ?Ùÿ?Þÿ?ãÿ?çÿ?êÿ?íÿ?ðÿ?r}?vä>ÝhÛ¾íu¿a¿¤­¾¬§>îcR?lô?ß)]?Û`?¾Ôy=k¤½¾Rº5¿hÉk¿Z½¿²¨u¿ZT¿ðb#¿ÑÓ¾å¤5¾U=ê>+æ>×U?<?jU?ÿh?XÛt?u|?ûº?Ño?oF|?Üv?»ºo?Tg?
^?Z,T?úI?(§??ú[5?µ7+?íQ!?÷»??¬?ú>ê>õVÛ>¦Í>G¿>ÉÏ²>îß¦>¨¬>f->TY>O|>;k>Å[>´L>¦>>Ü1>OÔ$>­y>dä>>x®÷=®æ=kÖ=ÃÇ=0ï¹=¨­=h¡=ùç=ä=©×=]­q=bê`=rPQ=ËB=FG5=4³(=Tþ=9=õ=°ý<t{ë<"#Û<íË<LÅ½<ð°<V¤<î<BP<o<Þzv<V^e<âqU<g F<AÖ8<,,<$ <Uó<
<|ü <÷ð;Feß;ËâÏ;÷sÁ;§´;§;ä;Ô;qÿ; @{;Îi;÷Y;êwJ;Wi<;T/;^(#;{Ô;íI;¹z;½³ô:¶ã:@çÓ:ú0Å:×>ö3e?rMg?×>H$ô¾¤íp¿æq¿QÙ¿­ù¼Ñí ?ðO\?û?µÊm?ÞN4?¼gÇ>­8=¾ù¿²E¿*i¿®ð{¿Ú¦¿Ãóv¿v¬d¿©|K¿1½-¿rb¿,õ×¾j¾Þ¹)¾ï;½Æ=Þ.>¼>£³>ìJÛ>xÒþ>+>?8K?¸)?Ã­4?"Q>? ÆF?§/N?ÙªT?TZ?YD_?8c?±Qg?j?|jm?aáo?fr?<ás??~u?äv?}x?)y?Þz?Ýz?5{?w%|?h©|?¼}?Ì~}?Ô}?ø~?`_~?+~?~Ç~?Zñ~??5?4P?Âg?(|?Ó??_ª?Ùµ?É¿?dÈ?ØÏ?LÖ?ãÛ?ºà?ëä?è?±ë?jî?Åð?Ðò?ô?ö?p÷?ø?ù?pú?/û?Ôû?cü?ßü?Ký?¨ý?øý?>þ?zþ?®þ?Üþ?ÿ?%ÿ?Bÿ?\ÿ?rÿ?ÿ?ÿ?¤ÿ?°ÿ?»ÿ?Äÿ?Ìÿ?Óÿ?Ùÿ?Þÿ?ãÿ?çÿ?êÿ?íÿ?Dy&?´í{?8Ò>Ûë¾\Üw¿B]¿4¾´>ùU?Wÿ?=QZ? Ñû>ü-=nëÅ¾e¢8¿Dm¿uà¿Í±t¿vR¿$!¿§Î¾¹¼+¾Ðòz=É>éÇé>rÔ?Q4=?APV?Àh?Du?~¬|?È?Õ[?¾|??v?ßgo?öf?	¥]?#ÂS?I?A:??ð4?Î*?4ë ?ÓX?ô"?©Q?Óù>Ûé>È»Ú>8rÌ>_ú¾> N²>3f¦>:>zÂ>Dõ>ç{>Woj>mkZ>9wK>]=>%}0>X$>O>ßx>n£>Ëóö=Ìáå=xûÕ=»,Ç=Àb¹=å¬=« =£v=c=tu=öp=5@`=²P=­7B=¾4=3(={=ª==Lü<'Éê<3}Ú<³RË<5½<5°<)Ú£<Fz<|ä<Ç
<3Àu<¡°d<;ÐT<ù	F<EJ8<ç~+<ê<<
3
<Ì <&Zï;¼Þ;YEÏ;sáÀ;O}³;,§;n;õ£;2;Õz;si;.îX;ÞI;¤Ú;;ÐÏ.;Ë¬";|a;êÞ;%;húó:
ã:ÂFÓ:èzB¿âæ5>9mi?@8c?!> ¿%Ts¿Co¿¬¿ñI;%±? å^?{Å?vl?¥T1?ÿ?À>÷"þ<Fy¾n¿ÜëF¿±8j¿À_|¿ã¿^v¿pÁc¿[VJ¿>r,¿=¿Ó2Õ¾ú´¾$¾~\(½ªÞ=x92>¤z>Vµ>ïÕÜ>¢ ?Û?&Ö?03*?Æ5?°>?ÒG?wxN?|êT?Z?Èt_?h¼c?jvg?{´j?@m?ùo?Yr?lós?u?Oòv?]'x?e3y?Ìz?Gåz?ë{?G+|?r®|? }?}?å×}?Î!~?Õa~?L~?VÉ~?óò~?ý?56?>Q?¨h?ï|??µ?áª?I¶?*À?¸È?!Ð?Ö?Ü?éà?å?°è?Ðë?î?Üð?äò?¦ô?+ö?|÷?¡ø?ù?yú?6û?Ûû?iü?äü?Oý?«ý?ûý?@þ?|þ?°þ?Ýþ?ÿ?&ÿ?Cÿ?]ÿ?rÿ?ÿ?ÿ?¤ÿ?°ÿ?»ÿ?Äÿ?Ìÿ?Óÿ?Ùÿ?Þÿ?ãÿ?çÿ?êÿ?h¾)÷:?)èu?¼¤>gº	¿qR|¿
~R¿AYw¾60Ô>LP^?	?JÉR?§å>Û»DÚ¾Ä¨?¿.¸p¿xÿ¿Rr¿½N¿Iu¿ÛDÂ¾µþ¾ä÷«=>¾¶ò>É ?É@?±X?áLj?a@v?û-}?ä?n$?§{?líu?n?	f?¬¥\?¥·R?Å}H?µ)>?iâ3?ÝÅ)?¨ê?Ta?5?ên?6$ø>mBè>ê8Ù>6Ë>¨¢½> ±>Ü6¥>>¸>øû>­Áy>ª»h>ÔX>fûI>õ<>2/>P$#>øæ>m>ç©>Ý"õ=¹0ä=hÔ=µÅ=ý¸==F«=zh=b\=¢=Þ=3/o=d^='O=È@=\h3={õ&=_=*=<=ônú<é<ößØ<"ÒÉ<¶Ï»<)Æ®<:¤¢<ÙY<Ø< <Sïs< c<¨=S<WD<¥í6<{:*<i<i<-	<Oÿ;Zí;¸Ý;=½Í;t¿;Á)²;1Ë¥;xH;T;;í§x;rdg;ËSW;¬`H;Cw:;-;y!;C;kÔ;&;Ú,ò:\á:,)u¿ß.¿VY>mr?6ËW?ô,> ³¿Ykx¿ûh¿ßý¾¡Ö=®F?ðd?Xÿ?¨g?´¶)?0>®>=è»ºZ¦¾à¿¤dK¿4Ûl¿@Y}¿¿J×t¿Lha¿;mG¿1)¿B¿jKÎ¾ò¾8¾Ö3ï¼N§=Æ<>ô4>¢¹>ïªà>Ï?D`?Ý.?}c+?®%6?W??èG?ÿ,O??U?J[?Ôì_?ú$d?mÑg?¤k?Ëm?H5p?CNr?} t?#µu?<w?ÊDx?éLy?ì1z?tøz?¤{?¯9|?íº|?ë*}?ó}?à}?×(~?íg~?~?éÍ~?èö~?k?-9?ÐS?âj?Ý~?+?'?"¬?_·?Á?É?ÕÐ?(×?¡Ü?_á?zå?é?ì?Æî?ñ?ó?Ðô?Pö?÷?¼ø?¶ù?ú?Hû?êû?vü?ðü?Yý?´ý?þ?Gþ?þ?µþ?âþ?ÿ?)ÿ?Fÿ?_ÿ?uÿ?ÿ?ÿ?¥ÿ?²ÿ?¼ÿ?Åÿ?Íÿ?Ôÿ?Úÿ?ßÿ?ãÿ?çÿ?%v¿Eiä½!U?±Ùg?«+9>*%¿ÒÀ¿@¿mØ	¾´ ??Hi?D}?z:F?ã­Á>tê½ÉBø¾ª°I¿ÖJu¿H¿¶m¿ãF¿«¿hF¯¾ÆÚ½(ò=õ"¥>¾ ?ý%?/mD?C»[?l?§w?¿Ý}?èû?RÁ~?]Úz?[Þt?YOm?d?Ç[?Q?ÞF?£<?JG2?³3(?nd?+é?ÞÌ?±?õ>êÕå>MîÖ>ãÛÈ>»»>a!¯>Mk£>Co>$>m>«ÿv>ú'f>xlV>Y¼G>j:>#>->³Q!>4>·×>30>!cò=8¡á=Ò=|Ã=óµ=ZY©==0±=õ=`~=~l=õ\=ÒL=Þ>=3c1=*%==Fö=B
=Ò÷<mæ<nÖ<Ç<²¹<Î¬<*Ï <W¥<ÞA<ü<Å/q<Nq`<bÜP<]\B<Þ4<}O(< <dÀ<î¡<nü;èê;Ú;ÏkË;VL½;Þ'°;ûì£;u;6ô;3;´Úu;Éd;´æT;´F;k]8;+;K§;°;A
;Ó§ ;Rrï:6â¾"g~¿«2¿Ù>Ç{?1CC?ËÏ3=:J)¿~«}¿-]¿ùÛÒ¾UQ>þ!?úl?©F?[ä_?O©?a>§i½ë¾¾@.!¿ãÙQ¿4p¿~¿D4~¿§Qr¿ó¯]¿ôäB¿-,$¿ÒZ¿ÁÂÃ¾Ñ©¾Ðï¾*7¼Í¸É=NL>BV>¤
À>pæ>b?µ©?5!?È,-?%¸7?Àü@?ØI?<P?
uV?îã[?÷ `?ÞÁd?÷Yh?czk?E2n?òp? r?dt?Êïu?Gw?ìpx?.sy?Sz?6{?w½{?IO|?¥Í|?#;}? }?0ì}?b3~?q~?|¦~?ÃÔ~?Øü~??¡=?«W?9n?Â?¬?S¡?®? ¹?Â?ÁÊ?äÑ?Ø?lÝ?â?æ?é?ì?)ï?kñ?_ó?õ?ö?Í÷?æø?Úù?­ú?cû?ü?ü?ý?hý?Áý?þ?Qþ?þ?½þ?èþ?ÿ?.ÿ?Jÿ?cÿ?xÿ?ÿ?ÿ?§ÿ?³ÿ?¾ÿ?Çÿ?Îÿ?Õÿ?Ûÿ?àÿ?äÿ?³@@¿É]¿Ïfö=uÜn?zzM?Ã±»¦æD¿k6~¿Eg$¿¿°;7Ô?Yt?ì*w?·4?ìa>h)¾Å¨¿U¿Tz¿ñü}¿0g¿r =¿÷×¿Ic¾ãÊr½M&>
¹>;t?ÊÚ,?ÈI?²_?ÏEo?Ny?P~?<û?&~?FÆy?tms?Ðk?#µb?Y?ôO?n¾D?k:?/00?)&?ýi??ºù	?'Y?ËFò>©³â>4øÓ>Æ>þø¸>;©¬>¡>BC>ø>X>ðps>Õb>ëQS>ÁÖD>S7>Û¸*>ø>5>Í	>BI >â×î=,TÞ=òÎ=µÀ=2I³=ûÝ¦=N==X=cz=Äi=&ÜX=ÑI= Ð;=È.=Â§"=)^=ëÜ=+=úó<Ö
ã<jHÓ<êÄ<ð÷¶<lDª<r<³r<G6<x_<ê¤m<f%]<ËM<?<ó52< Ö%<-S<Y<î£<n¹ø;Êtç;c×;ènÈ;º;|­;¡;ÜM;oÞ;{(;=>r;¼la;&ÆQ;»5C;*¨5;|);öN;	c;99;ý:
)?­¿ò#~¿<0¸¾õ°?
ÿ?l#?ñ½¼;D¿ÿ¿VJ¿µ¾,T>î5?mPu?«v|?ëT?´B?
<\>! ¾¼äÛ¾«,¿ Y¿.µt¿Ä¿%|¿³n¿
X¿ÒÔ<¿Ä¿×÷ø¾
¶¾ª§h¾dÕ½]E<½õ=ÿs`>	{>BHÈ>Òí>­?°?ÇÊ#?dt/?¹9?k¿B?²¨J?fQ?¿¢W?*ë\?fa?©e?Ñi?l?ªµn?q?;ÿr?ºt?n:v?Þw?©x?ß£y?P}z?Î9{?.Ý{?Åj|?vå|?ÅO}?á«}?­û}?Í@~?¯|~?°~?{Ý~?e?&?KC?\?yr?p?Ü?¤?g°?»?NÄ?NÌ?<Ó?<Ù?nÞ?îâ?Óæ?3ê?í?§ï?Øñ?¾ó?bõ?Îö?
ø?ù?ú?Ôú?û?ü?¥ü?ý?{ý?Òý?þ?]þ?þ?Æþ?ðþ?ÿ?4ÿ?Pÿ?gÿ?|ÿ?ÿ?ÿ?ªÿ?¶ÿ?Àÿ?Èÿ?Ðÿ?Öÿ?Üÿ?áÿ?iy>±k¿Ù -¿ÎòÅ>÷m~?Ñm#?âg¾ºYb¿R¶s¿¨xý¾Ê¨/>ØA:?Í|?
ýk?Bø?	¯0>Aö¾KY#¿ËÍa¿LÜ}¿èz¿6^¿#b0¿kð¾^p¾áÂ'»E3[>>Ð>©6?V»4?ËÜO?X!d?¹@r?¯{?
J?_Ì?ÈF}?8\x?Hq?yi?S_`?ÈV?Ï{L?¾,B?yÜ7?D¬-?lµ#?­
?p¹?×Ê?*þ>VSî>öÞ>»oÐ>¢¼Â>×µ>ç·©>bW>­>ý¯>iW>Ã4o>Fà^>G O>:dA>94>¹'>=,>h>Z_>øü=¤ ê=ôfÚ=sJË=ô7½=~°=@ê£==û=¤#=ûóu=?åd=ÅU=½=F=Ã|8=Â¯+=âÅ=r¯=Ò]
=fÃ =ü¦ï<ß<©Ï<°!Á<¹³<º?§<~£<Õ<Ç<|Øz<ani<»9Y<%J<e<< /<óå"<Á<<bE<Pô;HZã;zÓ;4áÄ;ý5·;µ}ª;§;«£;c;u³;Ãòm;m];
N;¹¿?;¬o2;;&;ú;Ê;ðÎ;¿}?\`È>Æ<¿êl¿Tâ½­E?/^y?i-ï>¸¾}m^¿4|¿ /¿Û_!¾ÖrÆ>F K?)|?dv?±E?8<ñ>O>f<R¾ê8þ¾9¿«øa¿íÕx¿Éÿ¿ßz¿cÞi¿#R¿rN5¿øm¿³Pè¾6¥¾­·H¾½"=ø>ãKx>\H©>ï Ò>'ö>À?\?iÔ&?Ó!2?;<?3ÐD?ÎwL?\*S?cY?£^?4b?³sf?pÓi?Âl?Oo?·q?Ass?Ìu?Îv?©Ów?Îêx?ÝÜy?¸®z?¡d{?L|?î|?T}?ëg}?ÌÀ}?Ì~?P~?H~?U¼~?¯ç~?<?Á-?ìI?Qb?rw?¾??P§?3³?~½?gÆ?Î?ÏÔ?Ú?ß?óã?¶ç?÷ê?Éí?:ð?Wò?,ô?Âõ?!÷?Rø?Yù?=ú?û?®û?Bü?Ãü?2ý?ý?åý?.þ?lþ?¢þ?Ñþ?úþ?ÿ?;ÿ?Vÿ?mÿ?ÿ?ÿ? ÿ?­ÿ?¸ÿ?Âÿ?Êÿ?Ñÿ?Øÿ?Ýÿ?È¶i?Aq¾0Ê¿êË¾â&?¨Ö{?ù\Ñ>Ì¡ì¾Köw¿pî\¿ýj¾Ð"µ>E$V?ùþ?ò-Z?­cû>;g)=ÇOÆ¾uÅ8¿9Vm¿Íá¿Ì¥t¿ÜR¿!¿ÚjÎ¾D+¾¹|=ÉO>Èóé>æ?²B=?[V?·Çh?óHu?¯|?:É?ÙZ?|?v?êco?
òf?1 ]?½S?kI?5??ûê4?ùÈ*?Qæ ?T?m?VM?fËù>¶Óé>g´Ú>BkÌ>Ðó¾>öG²>i`¦>5>e½>ð>{>gj>©cZ>ùoK>{=>Öv0>¬R$>Ó >Ãs>ª>ëêö=Ùå=ÅóÕ=%Ç=\¹=®¬=â =@q=_=Èp=×íp=8`=ªP=«0B=·4=o-(=Õ=\¥=²=ýBü<­Àê<PuÚ<\KË<Ã.½<Ù°<>Ô£<Ät<\ß<<T·u<_¨d<ÈT<ÒF<C8<¶x+<'<)}<.
<' <Qï;	´Þ;Ý=Ï;|ÚÀ;Ôv³;$§;æh;¼;V;Éxz;i;YæX;G×I;ÛÓ;;É.;ë¦";\;ÔÙ;"ðÐ>ÛÆx?àó%=£åj¿  B¿¹â7>mi?±c?x~>S¿ps¿«qo¿xH¿¸·;¶ê?_?íÇ?hl?01?£è¿>w¨ø<FÇ¾©¬¿G¿õEj¿àd|¿#¿cWv¿-¶c¿KHJ¿ub,¿ô¿3Õ¾¾SI$¾Hn'½QM=Kl2>Æ>|kµ>²èÜ>" ?uâ?¾Ü?9*?ç5?´>?ÅG?ì{N?íT?<Z?w_?i¾c?(xg? ¶j?m?¥úo?Xr?Iôs?Çu?õòv?í'x?â3y?9z?¥åz?<{?+|?¯®|?O }?Â}?Ø}?ñ!~?óa~?f~?mÉ~?ó~??C6?JQ?³h?ù|??¼?çª?N¶?/À?¼È?$Ð?Ö?Ü?ìà?å?±è?Ñë?î?Ýð?åò?¦ô?,ö?}÷?¡ø?ù?yú?7û?Ûû?iü?äü?Oý?«ý?ûý?Aþ?}þ?°þ?Ýþ?ÿ?&ÿ?Cÿ?]ÿ?sÿ?ÿ?ÿ?¤ÿ?°ÿ?»ÿ?Äÿ?Ìÿ?Óÿ?Ùÿ?/V?)p#?¾Ó¿ño¿(
#½é^?t¤`?ý=Ø/¿´þ¿éê7¿U:»½~ö	?Øm?Ê{?³×@?z6³>3Ó½ø¿ì{M¿Dåv¿z7¿ Ëk¿æC¿±¿§¾ª¼½Ì>ÈR«>¨?8#(?ÑF?Jû\?Øim?N/x?~?{ÿ?¤~?§z?znt?×Él?Ed?Ô}Z?*zP?£6F?bâ;?:¢1?f'?
È?R?<?÷?Âô>ÏÝä>ÐÖ>ÁþÇ>Éº>Ç]®>¼³¢>1Ã>w>«ë>Ååu> e>vU>ÇÖF>:19>_v,>q >d>ê5>r>?Jñ=à=õÑ=¯Â=T µ=¢¨=sæ=°=>ç=æõ|=kk=ö[=LäK=¿==Î0=T$=Èì=æO=lo= |ö<Ý`å<ôtÕ<Ö£Æ<ëÙ¸<õ¬<÷ <"÷<À< <üp<l_<;éO<zA<v4<'<Ãé<´<<¹Hû;Öé;Ù;ú~Ê;òo¼;ÇZ¯;!.£;ÜÙ;ñN;g;x¼t;§¾c;ÕîS;8E;Å7;ÜÈ*;kí;·ä;ü7¿¼	E?wI?J{²¾Ì¿é0ý¾Óõ>þ	~?õ:?&4Î»ê2¿í~¿Ð¤W¿ê%Á¾6°<>-^(?7Îo?¡~?É\?¯?T>W ½4`Ç¾ÁÌ$¿pRT¿ äq¿Ré~¿Ã}¿'>q¿º%\¿A¿#"¿=¿w¿¾i{¾ò)ø½Cw»¹\×=uR>},>tÂ>ìºè>9h??·"?0â-?ÂW8?àA?¡I?m§P?áÒV?Ï5\?Vè`? e?h?l©k?'[n?v²p?öºr?Ø~t?v?D[w?ex?Uy?<`z? {?VÇ{?ÖW|?Õ|?A}?}?ñ}?7~?­t~?©~?z×~?2ÿ~?!?d??2Y?o?ç?ª?/¢?Á®?¤¹?Ã?=Ë?OÒ?oØ?¼Ý?Tâ?Næ?Àé?»ì?Pï?ñ?}ó?*õ?ö?à÷?÷ø?èù?¹ú?nû?ü?ü?ý?ný?Æý?þ?Uþ?þ?Àþ?ëþ?ÿ?0ÿ?Lÿ?dÿ?yÿ?ÿ?ÿ?¨ÿ?´ÿ?¾ÿ?Çÿ?Ïÿ?Õÿ?¼¦?.M>Lf¿MÌ5¿4L²>¢'}?]*?GF¾¡^¿Þu¿¦¿}r>£!6?óÞ{?Qïm?'½?á5B>ö¾\ ¿`¿kh}¿*{¿»¢_¿-S2¿Çõ¾py¾6Ë4¼NS>TÎÌ>cÆ?k3?ùN?Í|c?fÓq?»Êz?À3?Ö?ìj}?x?âq?Ìi?÷¹`?UþV?ûÝL?ºB?Ù>8?ò.?$?·e?ñ??(ÿ>°êî>Dß> ÷Ð><Ã>ëN¶>(ª>Á>ú>¢>®>¾Öo>w_>-P>èA>54>¿+(>(>ªË>8¼>¹ü=ÕAë=ýÚ=DÖË=&º½=¸°=[¤=÷=[]=³~=v=e=¡U=nÆF=üû8=*&,=4 = =D½
=9=PLð<vß<ØÐ<ï¦Á<5´< ³§<á<x9<$<{<sj<ÏY<°J<2<<ë/<[V#<[ÿ<äq<÷<#ùô;+÷ã;x#Ô;iÅ;i´·;[óª;û;	;jÂ;ô1;õn;c^;:N;
D@;Íê2;Ð~&;ï;oý¿ÙU½ôÏz?|Þ>¼<4¿Üùo¿q>¾??³'{?@ý>³¾óZ¿Ñ8}¿æ3¿ ,7¾hì¼>yH?Z{?Ñ{w?-G?÷>Z>F¾ñ.ù¾¬7¿|È`¿Gx¿ü¿}z¿g j¿|!S¿s6¿¨¿¢Òê¾ä ¨¾6M¾~¡½ú¨=&>¾t>·¬§>·Ð>	:õ>_ù
?®?î`&?¼1?Çº;?»D?3L?IîR?ïÏX?ãñ]?Wkb?ÿPf??µi?\¨l?´8o?çrq?bs?ßu?Úv?mÈw?áx?jÔy?e§z?H^{?Ëü{?*|?3ý|?Wd}?²½}?~?-N~?D~?º~?,æ~?ì?,?ñH?xa?µv??	?Ö¦?É²?"½?Æ?ÛÍ?Ô?eÚ?oß?Íã?ç?Úê?°í?$ð?Dò?ô?´õ?÷?Gø?Pù?5ú?üú?¨û?=ü?¾ü?.ý?ý?ãý?+þ?jþ? þ?Ðþ?øþ?ÿ?:ÿ?Uÿ?lÿ?ÿ?ÿ? ÿ?­ÿ?¸ÿ?Áÿ?Êÿ?Ñÿ?ç¡X¿< ?FE`?e·¾à¿.Ç«¾¦2?9Ãx?ñ·>d¿¨z¿æW¿¨¾.úÆ>ÕèZ?oÐ?·V?ù¼î>x²x<ÁÑ¾ö½<¿Oo¿<ü¿¡:s¿Ù P¿Ý¿~Ç¾æi¾Ýk=&&>½÷î>ªø?æ>?ÐW?§i?¯×u?Ïø|?¥Ù?¶<?lÒ{?g6v?íîn?µmf?]?(S?aðH?>?T4?5*?±V ?É?x?[Î?ÈÙø>îè>¶ÛÙ>ÍË>G3¾>ô±>|¶¥>À>'(>Ød>Ïz>ñri>³Y>,J>
µ<>¡½/>ù¥#>Ö_>¹Ý>Ý>næõ=ãæä=ÈÕ=SÆ=¸=8Ï«=è=Ó='=¿ç=¹îo=¨J_=ÍO=öbA=ø3=@{'= Ü==ú=¦7û<âÇé<ÇÙ<äsÊ<>f¼<?R¯<&£<+Ó<úH<z<Ý²t<ýµc<ýæS<ë0E<I7<ðÂ*< è<Àß<	<× <ÏSî;ôÇÝ;+bÎ;À;¸²;P¦;'Ä;l;«;Coy;ñh;i X;OI;¾;;6.;~ú!;¿g¿¦U¿èãö>cs?e}ÿ¼t)q¿*e7¿Äq>èn?§Ð\?x!P>ñÆ
¿únv¿àk¿Z¸¿­=t?ewb?sø?ßi?Àó,?DÙµ> /<û±¾Ä;¿!I¿Äk¿õ|¿?J¿Qu¿ågb¿Á¨H¿*¿Î

¿A4Ñ¾Ë¾¾ ½ü=f?8>F8>TÐ·>Éß>?Å¼??³ã*?/µ5?8??~G?4áN?FU?|ÛZ?oº_?ùc?8«g?jâj?-®m?0p?y8r?t?¹¤u?ÿw?p8x?3By?£(z?hðz?{?£3|?°µ|?a&}?}?Ü}?ã%~?_e~?\~?ýË~??õ~?û?î7?¼R?ói?~?w??«?ê¶?¶À?1É?Ð?æÖ?hÜ?-á?Oå?ãè?üë?«î?ýð? ó?¾ô?Aö?÷?±ø?¬ù?ú?Aû?äû?qü?ëü?Uý?°ý? þ?Dþ?þ?³þ?àþ?ÿ?(ÿ?Eÿ?^ÿ?tÿ?ÿ?ÿ?¥ÿ?±ÿ?¼ÿ?Åÿ?Íÿ?üÓg¿ý«¾@Vo?ÌAñ>j®@¿Å\¿ßú=o?õ/M?1©ë»¢+E¿{*~¿$¿UHÜ;:?ot?w?Jì3?tí>q}*¾sÔ¿U¿z¿ ÷}¿g¿¬ç<¿êº¿~&¾Gòp½û&>Ò3¹>g?(ë,?;ÕI?j»_?*Lo?ÛQy?ì~?û?å$~?Ãy?Üis?k?w°b?Y?ÔO??¹D?×e:?+0?!$&?.e?Þü?Kõ	?íT?Ä>ò>¬â>ñÓ>N
Æ>ò¸>?£¬>|¡>þ=>
>¼>Qhs>Íb>fJS>¼ÏD>ûL7>¿²*>^ò>æý>&È	>§D >LÏî=-LÞ=*ëÎ=ÅÀ=½B³=ø×¦=yH=M=~=Zz=]ýh=SÔX=»ÉI=ÙÉ;=HÂ.=ä¡"=³X=Õ×=p=6ñó<¤ã<Ê@Ó<ÑÄ<Uñ¶<F>ª<Öl<`m<R1<@V<Um<j]<¡ÃM<¤z?</2<£Ð%<M<)<<r°ø;nlç;F[×;¬gÈ;Í}º;8­;Á~¡;nH;bÙ;È#;~5r;da;¾Q;®.C;¡5;a);þ-Ù>Åq¿³µ¾NÌa?k(?N¿t~¿ê5·¾
?Nþ?BG#?Îµô½ÆwD¿þ¿Û$J¿%¾¦Û>ì6?Æau?Gm|?sT?Ü?[>MÉ ¾þ+Ü¾0Æ,¿­Y¿¾t¿¿¬|¿Bªn¿SX¿ÖÅ<¿}z¿AÖø¾¦èµ¾Ùfh¾0Ô½{2I<(ö=±¤`>>&\È>îãí>û´?º?Ñ#?áy/?c¾9?¨ÃB?h¬J?¥Q?¥W?£í\?a?e?s	i?ýl?æ¶n?(q?) s?èºt?";v?zw?©x?T¤y?¶}z?&:{?zÝ{?k|?¯å|?÷O}?¬}?Òû}?í@~?Ë|~?¦°~?Ý~?x?*&?YC?\?r?y?ä?¤?l°?»?SÄ?RÌ??Ó??Ù?pÞ?ðâ?Õæ?5ê? í?¨ï?Ùñ?¾ó?cõ?Ïö?
ø?ù?ú?Õú?û? ü?¥ü?ý?|ý?Òý?þ?^þ?þ?Æþ?ðþ?ÿ?4ÿ?Pÿ?gÿ?|ÿ?ÿ?ÿ?ªÿ?¶ÿ?Àÿ?Èÿ?8¾d¾t¿ßº«>ümv?]ñÐ½3p|¿ê¨ ¿Xª?~5?Ë«ü>°Ã¾çq¿H)f¿àÀ¾>ØM?d¨?#a?°?:³=ÄØ±¾å1¿i¿¢u¿2óv¿÷ÈV¿:&¿
Ú¾BC¾6¸ =[ø>Ï	á>x5?S:?"T?3g?(Dt?é$|?)¥?Ú?|?1:w?{.p?HØg?=^?¯ÁT?ÂJ?@@?øó5?¼Ì+?Îâ!?ÖG?_?×,?Ëtû>7gë>2Ü>sÓÍ>GÀ>î³>ã§>éM>tÄ>´æ>wW}>Vl>õ[>)çL>²Ù>>e½1>.%>²>S|>0>B¶ø=gç=G×=¬È=¥µº=pÇ­=r½¡=!=ü=rb=»¯r=ßÚa=H0R=oC=,	6=ªg)=G¦=µ=ý=qþ<mwì<Ü<WÇÌ<`¾<íU±<{¥<À<è<Îü<©w<ÒSf<SVV<ütG<9<D¹,<t» <À<^0<<èñ;`Tà;KÁÐ;CÂ;UÆ´;Z9¨;p;­;î;
M|;ÆÈj;×{Z;PK; 3=;D0;x¿}?J'¾¥+q¿¯>	ª~?%8*>ûQ]¿´4S¿Jà =º§^?Ql?p§>¬(à¾1m¿ãèt¿ª2¿W¸S½Ã0ô>ÊX?õ?Ep?2o8?MpÑ>N=Âî¾ÌK¿ÏoB¿g¿I{¿Í¿Ä¾w¿Tòe¿¿M¿Q/¿mM¿ØÛ¾a;¾y1¾$W½ÁÖu=#(>Äâ>ç/±>¡Ù>ÖÞü> `?d?´
)?±4?¥Ê=?´PF?oÈM?¢PT?MZ?®ÿ^?fVc?g?@gj?Cm?+¿o?±çq?rÇs?Ýgu?2Ñv?¥
x?}y?5z?Òz?²{?8|?C¢|?}?ny}?ùÏ}?ñ~?ã[~?%~?àÄ~?ï~?¥?O3?»N?{f?{?Þ?K?¨©?:µ??¿?íÇ?qÏ?óÕ?Û?wà?±ä?Zè?ë?Dî?¥ð?³ò?|ô?ö?]÷?ø?ù?dú?%û?Ëû?\ü?Ùü?Eý?£ý?ôý?:þ?wþ?¬þ?Ùþ?ÿ?#ÿ?Aÿ?Zÿ?pÿ?ÿ?ÿ?£ÿ?¯ÿ?ºÿ?Ãÿ?7C?iN¿r ¿¯­\?u?z&¿Ml¿îJu»¡b?Ò²\?HÔÁ=ÜÏ4¿ß¿<²3¿»½ùN?°Ýn?¼¿z?>?å«>_Qð½÷Ü¿WO¿¿¦w¿ký~¿øËj¿`B¿h3¿`Â£¾#­½3
>0j®>æ÷?5)?²ñF?É]?ùÖm?²qx?a;~?ÿÿ?$}~??]z?5t?>l?»c?¬-Z?!'P?gâE?E;?CO1?ZA'?|y?Ü?ô
?ÖG?]ô>Qaä>*Õ>ÒÇ>.aº>ªû­>ªW¢>äl>§2> >hXu>d>-ûT>©cF>ÓÅ8>4,>: >V/>Êä
>ÛM>i½ð=hà=ÉÐ=é'Â=d¶´= 2¨= =4±=«=Àa|=#áj=Z=ÕlK=ëO==R-0=Äó#=&={ü=É!=!ìõ<dÚä<Ï÷Ô<`/Æ<m¸< «<¶<Ê<xN<{´<5o<é^<QoO<¤A<ã¡3<K)'<V< Á<Ú´<`µú;xMé;êÙ;<Ê;r¼;óó®;qÎ¢;Ð;ü;K2;ô,t;9c;rS;eÄD;'7;%?m?0©]¿ïÁ¿}'L?'ÐB?áñÄ¾ÿ¿ðî¾2¹?ÓÙ~?û95?5Ü ½5U6¿¦`¿ÍT¿Û)¸¾VN>^+?$q?>;~?ÒZ?E'?b°>BÀµ½<Ì¾ö&¿¤U¿rr¿A¿v}¿6°p¿][¿@@¿Ì!¿- ¿Ùf½¾8óv¾ Hð½<Ä¹2Þ=®U>>8áÃ>àé>'ë?Z?i"?æ<.?§8?ïÎA?ÖI?ÝP?ÉW?¼^\?a?e?«h?îÀk?on?5Äp?_Êr?8t?v?Uew?!x?çy?Ìfz?J&{?DÌ{?\|?ÂØ|?ÄD}?X¢}?jó}?¥9~?|v~?/«~?ÕØ~?^ ?"?F@?öY?5p?y?)?¢? ¯?÷¹?ZÃ?zË?Ò?Ø?äÝ?wâ?læ?Úé?Òì?dï?ñ?ó?7õ?©ö?é÷?ÿø?ïù?¿ú?tû?ü?ü?ý?qý?Éý?þ?Wþ?þ?Áþ?ìþ?ÿ?1ÿ?Mÿ?eÿ?zÿ?ÿ?ÿ?©ÿ?´ÿ?¿ÿ?Õt?Çï»Ò{¿ÖØl>`+|?Ý3¼NÀw¿>¿-A ?Ûý?P«?²p¦¾Ál¿÷Ãk¿º×¾Êõ|>ÎbF?î~?v\e?­õ?¾3ô=Û|£¾],¿½Óf¿Ñü~¿¸dx¿R¡Y¿Ã?*¿ºÍâ¾fT¾¢úÁ<æs>ÑÚ>Å?«C8?UR?f?âs?>¿{?ä?Ê¥?ÔÐ|?|¨w?á·p?Luh?ªC_?rtU?ÆIK?ø@?{ª6?Ë,?ö"?
ð?å©?7Ç?³ü>~ì>é9Ý>yÌÎ>¢1Á>c´>[¨>)>nz>>	~>?m>§]>ÅêM>ôË?>_2>çU&>$á>j3>Ë?>'ôù=­è=Ø=É=Ð¤»= ¦®=Æ¢=%I=ªÌ=´	=çs=²üb=>S=D=Ûò6=&A*=°p=èq=M6	=¿`ÿ<§í<9(Ý<UÎÍ<¿<°9²<pÚ¥<ÿV< <§<Àx<«{g<£iW<0uH<:<!-<ê!<áR<)ã<ë,<Fò;tá;sÍÑ;<Ã;®µ;r©;T;:h;>;"};^ök;~[;ÊUL;	&>;8¾Aþ?¼=8¾äy¿æz0>Öý?ì>öR¿*]¿<cãU?r?ÄÂ>vÇ¾À*h¿àx¿Ì!¿³ãº½]jã>°S?p,~?+r?G=?;gÝ> öµ=¶Ãw¾ÅÎ¿/?¿9e¿°nz¿ í¿?§x¿Fqg¿ÿN¿·1¿/¿Éà¾Ò¾ðà9¾Bæx½ÖV=Ô!>£>Ù?®>ñwÖ>	ú>qS?i?8(?[3?ö'=?
ÂE?KM?wãS? ¦Y?¬^? c?Þf?n0j?um?Ão?ºÃq?9¨s?ÅLu?±¹v?Aöw?Ïy?àöy?EÅz?-x{?<|?|?}?ðr}?XÊ}?~?ªW~?}~?µÁ~?Wì~?D?@1?óL?ðd?·y?µ?K?É¨?y´?¾?\Ç?ôÎ?Õ?8Û?&à?kä?è?Që?î?}ð?ò?^ô?íõ?G÷?rø?uù?Vú?û?Áû?Rü?Ñü?>ý?ý?îý?5þ?sþ?¨þ?Öþ?þþ?!ÿ??ÿ?Yÿ?oÿ?ÿ?ÿ?¢ÿ?®ÿ?¹ÿ?*´>U.L? >F¿r«
¿'V?õä"?øW¿fºo¿¶ò½ÈÄ^?äg`?ª]ù=a'0¿üý¿©7¿z}¸½l;
?2m?°{?I¬@?rÃ²>¿GÕ½é%¿jM¿kñv¿4¿w»k¿¦ÎC¿é¿ß`§¾s¼»½z>«>Ë¼?4(?)F?]?pm?m3x?û~?ÿ?7~?z?þjt?±Ål?§d?æxZ?uP?u1F?6Ý;?1?i'?4Ã?ÚM?8?¶?¬ô>%Öä>üÕ>í÷Ç>)Ãº>¼W®>®¢>á½>}~>ç>Ýu>ue>ënU>°ÏF>*9>4p,>± >>ì0>Ê>Añ=à=o
Ñ=­Â=Îµ=¨=Ìà=m=Xâ=Çì|=bk=[=ñÜK=B¸==o0=%N$=Cç=ÃJ=¤j=ºsö<Xå<?mÕ<ªÆ<?Ó¸<¿þ«</ <Áñ<¿<vû<Qp<òc_<¹áO<sA<õ4<}'<!ä<w<&ÿ<§?û;&Îé;©Ù;ªwÊ;$i¼;rT¯;<(£;`Ô;×I;¨z;¡³t;n¶c;.çS;ì0E;Wmv¿l?µù!? 0W¿i¿ß|E?¨'I?g ³¾²Ñ¿IEü¾óeö>a~?èÁ9?p ¼ÈV2¿õ~¿§xW¿ÇÀ¾ãÆ=>Ù(?­ão?§~?Çr\?K?Åû>þk¡½¡©Ç¾"é$¿¸eT¿"ïq¿ì~¿è¿}¿|5q¿m\¿cü@¿]"¿Ö,¿ôe¿¾8Õz¾¸­÷½Ø_v»~È×=Î¾R>äB>¦­Â>Íè>Jp?D?	"?Çç-?­\8?2A?jI?¼ªP?ÅÕV?U8\?ê`?ðe?·h?ßªk?i\n?³p?é»r?«t?»v?ã[w?ïx?Ìy?¤`z?ó {?£Ç{?X|?IÕ|?ÁA}?¼}?(ñ}?°7~?Êt~?·©~?×~?Dÿ~?©!?r??>Y?o?ð?²?5¢?Ç®?ª¹?Ã?@Ë?RÒ?qØ?¿Ý?Vâ?Pæ?Áé?¼ì?Qï?ñ?}ó?+õ?ö?à÷?÷ø?èù?¹ú?nû?ü?ü?	ý?ný?Æý?þ?Uþ?þ?Àþ?ëþ?ÿ?0ÿ?Lÿ?dÿ?yÿ?ÿ?ÿ?¨ÿ?´ÿ?éã)¿+Ðu?Cþ¡¼7E{¿iw>¶{?eB¼Mx¿ç¿ùë?ø?öD?2a©¾Mm¿½<k¿6sÕ¾`½>G?§?*ñd?C<?û»í=Bì¤¾äâ,¿,g¿
¿_Ax¿îYY¿ á)¿|ûá¾°»R¾n©Î<Çýt>gpÛ>ßá?xx8?¸R?n0f?Ãs?É{??.£?É|?¢w?Pªp?Äeh?È2_?¸bU?7K?Kæ@?[6?n,?¨"?Sß?×?Þ·?{}ü>Wbì>­Ý>µ³Î>NÁ>M´>kF¨>×ü>Th>>Yv~>s!m>þï\>òÐM>Ú³?>å2>ñ@&>Í>3!>Ó.>Ôù=è=£zØ=÷É=»=ù®=%x¢=ò5=Êº=ù="Ès=Ýßb=5#S=}D=Û6=+*=\=*_=Ü$	=I@ÿ<âí<Ý<,´Í<Æl¿<#²<ZÅ¥<`C<Ü< <ó x<=^g<?NW<³[H<Èr:<-<`u!<Ä?<`Ñ<^<Â'ò;ÞWá;Æ²Ñ;¼#Ã;rµ;òû¨;@;U;¾,;äp};]Øk;x[;Î;L;³?¿¹þ¾/ó?(íC¾[gx¿l:>úõ?35y>KS¿ñ\¿w<eÍV?îq?ÇÀ>üÉ¾D±h¿oÆw¿Þô ¿åÖ²½=å>£-T?õD~?÷Zr?gÍ<?+8Ü>|±=øy¾ÁA¿º?¿·e¿9z¿%ë¿·x¿¨Kg¿uÏN¿A1¿è^¿Jà¾h]¾k 9¾u½0ìY=·!>çõ>½®>>»Ö>/¿ú>3n? °?tM(? n3?-8=?BÐE?þWM?YîS?¯Y?Ý´^?8c?âäf?å5j?5m?äo?PÇq?V«s?xOu?¼v?Jøw?
y?gøy?Æz?Sy{?;|?y|?Ì}?s}?èÊ}?~?X~?Ú~?Â~?ì~??t1? M?e?Ùy?Ó?d?à¨?´?©¾?kÇ? Ï?Õ?AÛ?.à?rä?#è?Vë?î?ð?ò?aô?ðõ?I÷?tø?wù?Wú?û?Âû?Sü?Ñü??ý?ý?ïý?6þ?sþ?¨þ?×þ?þþ?!ÿ??ÿ?Yÿ?oÿ?ÿ?ÿ?¢ÿ?¯ÿ?¤ï|¿F³>²®??|WQ¿Vø¾Ù^?^6?'î(¿ïj¿I<d?:A[?mæ¬=¸6¿¨Ê¿¹.2¿§~½Ô?eyo?lz?p=?N©>!zú½á¿êüO¿éw¿*ç~¿pj¿ùÕA¿/¿Fe¢¾8¨½Ã>¯>&n?j)?Þ<G?kÑ]?4ým?Ýx?×E~?ùÿ?¬t~?!Nz?c!t?Fnl?z c?OZ?Â	P?ÄE?p;?ñ11?¸$'?½]?$ì?oÚ
?m/?÷Ýó>\5ä> dÕ>§hÇ>Q<º>Ù­>)7¢>mN> >_>&u>âmd>¤ÏT>;F>ê8>Øî+> >>'È
>,3>²ð=êß=©lÐ=ÁÿÁ=ÿ´=/¨=6j=={=u-|=v°j=MiZ=ªBK=¬(==Ë0=ÅÑ#=s=	ß=b= ¹õ<íªä<£ËÔ<EÆ<IG¸<|«<ú<ö<Ç1<È<Xo<Ãº^<IDO<à@<|3<'<n<£<è<]ú;é;àíØ;SÞÉ;qÚ»;¨Ï®;ª¬¢;ba;ÕÞ;;Lús;÷	c;°FS;ô>÷×o¿°)?Y?ïõ_¿ïûû¾µN? Y@?dË¾Zý¿Ô¹è¾ &??è}3?;%½ùÏ7¿Û¿ÔÆS¿ïù´¾T>b,?Kq?~?Ý3Z??A?%c> d½½©Í¾29'¿öU¿Èr¿"¿&p}¿}p¿Á[¿9Æ?¿Í¿ ¿oÿ¾a¦¼¾}u¾¨í½´©k:à=6ÉV>c>ÓTÄ>2Hê>R?W/?Ó"?â\.?³Ã8?¢çA?¦ëI?ðP?QW?)m\?a?	*e?´h?7Ék?Èvn?vÊp?ÍÏr?ït?¶v?âhw?4x?y?iz?K({?Î{?]|?Ú|?åE}?S£}?Cô}?b:~?w~?¼«~?OÙ~?È ?ù"?@?;Z?qp?­?V?Ã¢?B¯?º?sÃ?Ë?Ò?­Ø?òÝ?â?wæ?ãé?Úì?kï?¤ñ?ó?;õ?­ö?í÷?ù?òù?Áú?uû?ü?ü?ý?rý?Êý?þ?Wþ?þ?Âþ?ìþ?ÿ?1ÿ?Mÿ?eÿ?zÿ?ÿ?ÿ?©ÿ?ÞÎ¾¯à
¿Ur}?£­¾Õbs¿³>mXu?§Èí½4ú|¿¹Þû¾1?û~?Aø>®È¾ír¿ÔBe¿2½¾Jq>N?v»?zS`?{{?MC©=ôþ³¾VN2¿¦j¿¿Ñ¸v¿6YV¿<õ%¿îBÙ¾ÖA¾|P*=>µøá>?w¢:?þ]T?Ñ]g?á_t?Ë3|?X©?R?{|?X)w?p?xÀg?o^?¥¦T?wJ?¾$@?lØ5?¸±+?È!?x.?ð??~Hû>0=ë>S
Ü>í­Í>¶#À>³e³>¯l§>¥0>©>Í>z'}>}èk>®Ë[>ÀL>5µ>>\1>qc%>ÿ>À`>~{>bø=ÎXç=½X×=þqÈ= º=æ¥­=8¡=k=ìý=BI=Ör=9¯a=¨R=uC=ûå5=éF)=Ì=)=l=Méý<±Iì<ãÛ<»Ì<k¾<3±<æ¤<t<ìÌ<ã<ÈRw<D'f<Ý,V<gNG<-x9<Ú,<\ <Ñu<q<{m<Eâð;ú(à;éÐ;pÂ;\£´;Ï¨;'m;ë;µu;;|;[j;QZ;,j?IW¿xD¾}?;¼¾É»o¿?,>ÆD~?ä>Éá^¿cQ¿¶=®ä_?Q§k?f£>ªÑã¾æm¿Âct¿Æà¿YG;½D°ö>ø5Y?Ê?«¨o?N±7?ãÏ>D{=W¾÷¿ëB¿Ïg¿rh{¿RÇ¿w¿»·e¿ÍL¿d9/¿ô¿Å#Û¾)¾º/¾wR½©z=¹5)>ä_>¡±><Ù>9ý>^?)ª?K*)?04?ã=?#fF?2ÛM?aT?Z?*_?Fac?'g?|oj?FJm?cÅo?íq?"Ìs?ïku?ºÔv?µx?%y?z?Ôz?m{?·|?£|?¬}?hz}?ÑÐ}?­~?\~?²~?ZÅ~?ï~? ?3? O?·f?A{?
?r?É©?Wµ?X¿?È?Ï?Ö?¤Û?à?¼ä?cè?ë?Kî?ªð?¹ò?ô?ö?`÷?ø?ù?fú?'û?Íû?]ü?Úü?Fý?£ý?õý?;þ?wþ?¬þ?Úþ?ÿ?#ÿ?Aÿ?Zÿ?qÿ?ÿ?ÿ?£ÿ?M*?ºm¿¼?ºÈ?À1h¿exª¾+o?/ð>ÌA¿w\¿¿Vÿ=j<o?BåL?«Ú¼ypE¿d~¿©×#¿÷ç<+T?ìt?=w?Ó½3?øx>j\+¾ ¿µU¿Ýz¿Gò}¿g¿ãÎ<¿Ü¿²é¾ºo½£ñ&>c¹>?û,?íáI?ÀÄ_?Ro?¡Uy?~?áú?H#~?ÌÀy?Cfs?\k?Ë«b?Y?µüN?´D?«`:?&0?)&?^`?=ø?Ýð	?´P?½6ò>t¤â>ÖéÓ>Æ>?ì¸>B¬>Þ¡>º8>>>²_s>úÄb>áBS>¸ÈD>oF7>¤¬*>¬ì>ø>5Ã	>@ >·Æî=.DÞ=·ãÎ=ÖÀ=H<³=õÑ¦=áB==¥{=ýPz=õôh=ÌX=sÂI=Ã;=ú».="=<S=ÀÒ=µ=gèó<rúâ<)9Ó<¸Ä<ºê¶<!8ª<g<h<^,<M<Ám<n]<3¼M<ºs?<)2<¦Ê%<H<ú<H<x§ø;dç;S×;o`È;wº;ó­;ìx¡;C;VÔ;;¿,r;t\a;ªU?lÁ½clX¿Â$K?Ú×>Ðdq¿L´¾U b?(?º¿¥ ~¿o;¶¾øx?^ý?ýó"?DÔ÷½´³D¿àý¿óI¿¾c>ÉI6?su?×c|?5VT? ñ?ØZ>q¾8sÜ¾Eá,¿¿Y¿ÔÇt¿?¿0|¿ü n¿X¿Ù¶<¿6j¿¬´ø¾9Çµ¾&h¾ Ô½xÌL<ö=aÕ`>2§>pÈ>¼õí>ê¼?Å£?;×#?^/?6Ã9?åÇB?°J?äQ?j¨W?ð\?µa?je?i?hl?"¸n?:q?s?·»t?Õ;v?w?"ªx?É¤y?~z?~:{?ÇÝ{?Ik|?èå|?(P}?7¬}?÷û}?A~?ç|~?¾°~?¥Ý~??9&?fC?«\?r??ì?#¤?r°?»?WÄ?VÌ?BÓ?AÙ?sÞ?òâ?×æ?6ê?"í?©ï?Úñ?¿ó?dõ?Ðö?ø?ù?ú?Õú?û? ü?¥ü?ý?|ý?Òý?þ?^þ?þ?Æþ?ðþ?ÿ?4ÿ?Pÿ?gÿ?|ÿ?ÿ?ÿ?9ú?½K&¿ò¾n~?óÞÀ¾ok\¿R?F=c?!¾Ìü¿ANµ¾Æ2/?vÃy?¿>¾tü¾éy¿±ãX¿¾«´Á>WY?{å?HW?/ò>Z¼<^Î¾;¿õ»n¿÷¿¯¨s¿,ÁP¿5Ï¿9É¾!¾Ý®=Êj>{í>§[?Jj>?2:W?|ei?­­u?@ã|?Õ?íE?æ{?ÿRv?Öo?&f?ÿ;]?nTS?I?ðÉ>?î4?a*?d ?¹ò?üÀ?ô?!ù>¬2é>Ú>ÛË>{l¾>»È±>øè¥>Å>}T>V>gÓz>s»i>kÃY>cÚJ>ð<>¥ô/>FÙ#>©>J
>d<>Ï3ö=÷.å=éTÕ=Æ=RÒ¸=k¬=x ==â­=t=:p=1_=)P= A=ö04=.°'=B=ß8=º$=û<Éê<ÒÙ<å³Ê<Î¡¼<­¯<-Z£<+<¦u<¢£<; u<üýc<ý)T<EoE<Nº7<ïø*<?<<Ç	<U6 <+î;Þ;m£Î;ÅJÀ;ñ²;°¦;õ;÷2;0;"¾y;Vgh;·Y¼Ø¡B?`t¿ÚWâ=I$m?Y2¿¯X¿*Èë>Z$u? "¼sio¿­¨:¿¦`>hm?V·^?)û]>
ÿ¿Ôu¿¸øl¿çû¿é<V
?æua?ºî?­Hj?B8.?|Ø¸><N¾D.¿ÉH¿GTk¿yË|¿A\¿mÀu¿ïËb¿Â$I¿º+¿
¿gZÒ¾ë¾L0¾:½"8=£6>Ön>i·>kÞ>¶Î ?ü{?¸d?±*?5?^??)oG?(ÃN?Â+U?ÄZ?t¦_?­çc?g?=Õj?¹¢m?=p?Õ/r?t?6u?Y w?3x?ô=y?ô$z?6íz?Ì{?=1|?³|?$}?u}?AÛ}?¸$~?[d~?{~?;Ë~?ô~?i?o7?NR?i?¼}?0?N?f«?¼¶?À?É?kÐ?ÌÖ?RÜ?á?>å?Ôè?ïë? î?ôð?øò?·ô?:ö?÷?¬ø?¨ù?ú?>û?áû?nü?éü?Sý?¯ý?þý?Cþ?þ?²þ?ßþ?ÿ?'ÿ?Dÿ?^ÿ?sÿ?ÿ?ÿ?÷q?wüb>°k¿D?.¶>}|¿j£³½ºú}?;>¿þ^¿·%@¿è>ñÞz?I3?{¾¡Y¿ymx¿ff¿7Fé=.x0?rz?Kbp?B¯$?§Y>%up¾¸O¿ »]¿t¼|¿¯ï{¿}a¿ã4¿zßú¾IÖ¾=·¼±½H>2È>í×?_2?aÇM?Ub?â>q?»uz?ý?fâ?°}?£Þx?¶@r?X:j?B2a?N}W?`M?}C?àÁ8?Ã.?$?ß?£?*?Nýÿ>¡´ï>oDà>«Ñ>8æÃ>óî¶>Ò¾ª>N>>P>À!>¯p>A`>*êP>B>j;5>áÄ(>ì%>»P>78> ý=ì=ÅÛ=÷Ì=h¾=8±=Õñ¤=ã=úß=Jø=Üw=»Uf=½[V=õ|G=Ý¥9=FÄ,=;Ç =ó=·<=Õ=)ñ<êkà<
ØÐ<ÝXÂ<5Û´<7M¨<F<ë¾<¾ <²l|<æj<ÆZ<ÎjK<K=<7'0<tì#<	<ßó<í<CÚõ;©Èä;kæÔ;yÆ;;]¸;u«;,§;;A;Ã§;8ro;ç;Y¿£¡y?|È¾-õ#¿üAo?r{'>h¿Mg ¾Õ¸v?xû>0))¿¶t¿ôK¾-¼6?¯}?D?'7w¾§V¿{U~¿(u9¿V&T¾÷°>UÿC?P(z?×x?õ¼J?åÿ>wþ">ùº5¾µgò¾Ý&5¿µ)_¿w¿¡ï¿b{¿ok¿=qT¿ø7¿BI¿þ(î¾¦N«¾áS¾§­½y2ö<Ñà>ÿo>J¥>Î>ó>,4
?ÌÓ?]Æ%?Ø31?C;?­D?×K?ØR?±X?´]?ô5b?"f?Ïi?4l?'o?_Xq?Ks?àût?sv?`¹w?Ôx?Éy?z?ÈU{?mõ{?Ç|?«÷|?_}?¹}?~?K~?~??¸~?&ä~?+
?+? G?T`?¹u?@?L?1¦?:²?§¼?­Å?~Í?CÔ? Ú?3ß?ã?gç?³ê?í?ð?+ò?ô?¡õ?÷?9ø?Dù?+ú?óú? û?6ü?¸ü?)ý?ý?ßý?(þ?gþ?þ?Íþ?öþ?ÿ?9ÿ?Tÿ?kÿ?ÿ?ÿ?Â:Û¾j?õùd¿ó=Ä³e?z&¿Ý,¿ÐÂ>?©R=?sQ¿Iøx¿Ç¢¾P?k?1Z>|¿ST¿ósD¿HÔ"¾g
ø>)úf?~Ó}?Ó0I?éÎÉ>*_r½ñ¾ß{G¿St¿x½¿&Ân¿ýH¿æ¥¿Ü³¾Õë½XIâ=M¡>Qý>Â$?~vC?d[?Ùl?¬Xw?¸}?_ø?_Ù~?.{?äu?Mm?àîd?iv[?|Q?<G?9è<?¯¤2?(?þ¼?>?©?¾d?ü)ö>bæ>=s×>DYÉ>Â¼>M¯>hÓ£>ÚÐ>ü>ð×>w>`½f>ü÷V>>H>ê:>v¯->]»!>>3>º>{ó=¦5â=ÜÒ=RýÃ=zk¶=ôÈ©==ê= à=Å~=m=1¨\=¾YM=¯?=LØ1=%%=û=«T=b=T@ø<<ç<(üÖ<åÈ<·,º<>@­<^9¡<-<×<ì<Ïq<a<VfQ<½ÜB<~U5<©¾(<<ª <û<Zý;=ë;h)Û;-òË;_É½;9°;BY¤;8ð;úQ;tp;}v;_Xg¿ê:Ï>öä>(`¿Ñâ>¾\B?q=¿¹*¿íP,?Y\?ïSn¾ }¿4¿ÕÈ>üz?Ü<H?$=$$¿'¾|¿úó_¿¸ÆÜ¾*>QL?kFk?*?¢·a?q ?LÙ>8½~¸¸¾l¿lP¿ »o¿	L~¿ºn~¿rér¿	^¿ïC¿Q%¿¿'Æ¾ ÿ¾Àc¾ûùy¼¨úÁ=ËÁH>ª¹>¾>d"å>¹Í?%?À ?Å,?V]7?­@?ù×H?áþO?¢?V?Tµ[?Xx`?}d?-;h?_k? n?»zp?r?ÖTt?âu?¥;w?ùfx?jy?Kz?»{?Ù·{?jJ|?mÉ|?{7}?Õ}?qé}?1~? o~?´¤~?8Ó~?û~?g? <?ÍV?ym???Ö ?­?¢¸?2Â?{Ê?§Ñ?Ý×?>Ý?çá?ïå?né?tì?ï?Xñ?Oó?õ?{ö?Â÷?Ýø?Òù?¦ú?]û?üû?ü?ýü?eý?¾ý?þ?Oþ?þ?»þ?çþ?ÿ?-ÿ?Iÿ?bÿ?wÿ?ÿ?8å}¿ìï^?t¾vÛ+¿u?1¼B«{¿Bço>Û
|?h¾0¼íèw¿'Â¿º ?«ü?´E?F§¾	él¿¼k¿£×¾}>~>§F?8õ~?>e?Á?]ò=;å£¾,¿Yèf¿¿ ¿¶Zx¿Y¿%*¿
â¾UíS¾°Å<÷s>©þÚ>²?«R8?ÅR?Äf?@s?.Â{?Î?¥?ÆÎ|?h¥w?´p?äph?ß>_?joU?DK?`ó@?V¥6?¾z,?"?Kë?V¥?ÛÂ?gü>1vì>v2Ý>qÅÎ>+Á>J]´>+U¨>¬
>Ju>6>
~>¨6m>Ì]>pãM>Å?>ý2>óO&>Û>>.>ú:>,ëù='¥è=IØ=CÉ=»=Õ®=ê¢=±C=Ç=û=PÞs=ôb=l6S=jD=Aì6=;*=øj=l=Y1	=Wÿ<í<> Ý<çÆÍ<5~¿<@3²<sÔ¥<lQ<ï<Ë¢<·x<Osg<ÜaW<ómH<Ã:<Ý-<!<tM<Þ<8(<Ð=ò;dlá;àÅÑ;5Ã;ü§µ;W©;ØN;ñb;&9;ú};ê¾¨¬û¾²x?P½=¿µ¾&ü?ë;¾àx¿.\3>0ü?me>ûsR¿·G]¿Ú5%<&V?¯óq?ZÂ>Ï5È¾Qh¿Þûw¿m!¿¸½Îäã>ÀÓS?x3~?r?$=?4Ý>R´= dx¾pï¿ÖF?¿öe¿uz¿ðì¿Ü x¿fg¿óñN¿§1¿£¿$`à¾ú°¾.¡9¾aõw½Æ¶W=l5!>º>U®>Ö>ú>
[?%?u>(?Ê`3?,=?ÆE?OM?æS?³¨Y?î®^?c?càf?ü1j?Ïm?ïo?¿Äq?©s?Mu?[ºv?Õöw?O	y?O÷y?¥Åz?x{?|?Û|?C}?s}?Ê}?5~?ÉW~?~?ÌÁ~?kì~?U?O1? M?ûd?Áy?¾?R?Ð¨?´?¾?aÇ?÷Î?Õ?:Û?(à?mä?è?Rë?î?~ð?ò?_ô?îõ?H÷?sø?vù?Vú?û?Áû?Sü?Ñü?>ý?ý?ïý?6þ?sþ?¨þ?Öþ?þþ?!ÿ??ÿ?Yÿ?oÿ?ÿ?ê¾$¿Ë
>Ô»?ä¿?Ë["?UÌd¿â¸¾0#m?qû>º6=¿J/_¿	×=	m?O?zaå;7ÿB¿Î~¿ô[&¿Qs»Â?m¹s?;©w?â^5?y>#¾êu¿^ÁT¿ß¼y¿x!~¿L¦g¿°­=¿W£¿3¾»½?#>ä´·>°æ?îg,?RoI?tp_?!o?z3y?î~?`ü?·1~?cÙy?s?g·k?ØÕb?Â9Y?Õ*O?ÁâD?8:?ÛS0?èK&?³?ñ!?Ê
?Åv?ò>öèâ>*Ô>@Æ>¨%¹>3Ó¬>zA¡>)h>>>«»>_­s>c>£S>õE>n7>ªã*>ù>i(>Äï	>i >ï=:Þ=Ñ&Ï=RÐÀ=vv³=!§=Ou=
¯=Y§=Y¢z=°@i=ýY=J=# <=Îô.=éÐ"=v= =W7=Â7ô<KDã<ã}Ó<­ÏÄ<?&·<oª<©<<Y< <ám<`]]<&ÿM<²?<c2< &<=z<³À<ÂÅ<cøø;`¯ç;×;¥¡È;À³º;l½­;x­¡;çs;×;nI;-òC?õ}¿»¤O?sHí¼á]¿³íE?;¬å>àn¿YäÀ¾©_?o,?Ëú¾½~¿e¿¾jï?eþ?6Þ%?þ³Û½ûB¿ÿ¿°K¿D¥¾f>Øª4?RÕt?%·|?]U?y`?%a>$
÷½¤ðÙ¾µì+¿Y¿Yst¿$¿,»|¿3ôn¿ôòX¿¶==¿Ãü¿ãù¾Mô¶¾òmj¾øtØ½.X,<Óò=_>4à>Î¼Ç>>Uí>gu?Od?#?âM/?°9?²¡B?¨J? ~Q?ÛW?ÐÙ\?Fwa?~|e?\üh?l?­n?ùp?³ør?n´t?5v?w?a¥x?ª y?zz?e7{?Û{?õh|?äã|?iN}?³ª}?§ú}?ë?~?ë{~?ä¯~?èÜ~?æ?¬%?ìB?A\?1r?2?§?ç£?>°?ïº?0Ä?4Ì?%Ó?(Ù?]Þ?ßâ?Ææ?(ê?í?ï?Ññ?·ó?]õ?Êö?ø?ù?ú?Òú?û?ü?£ü?ý?zý?Ñý?þ?]þ?þ?Æþ?ðþ?ÿ?4ÿ?Oÿ?gÿ?{ÿ?½>Öd8¿ü-?T6¿a¾ì{?á¾ëyS¿Ð2?	;\?5­¾÷¿¢[¾ï6?\w?ª®>ï¾¿
{¿ÙÇT¿ª¾áÁÍ>«\?*¬?QbT?éÓé>½Â§;Ö¾×=>¿Æ	p¿«ÿ¿©r¿!O¿¸¢¿æÒÄ¾²¾½l¢=Èb>öâð>$Ã???MX?nüi?v?@}?-ß?w0?¸{?1v?Án?:f?gÚ\?îR?ÊµH?Øa>?ï4?9ü)?p ?<?Wf??æ|ø>è>mÙ>9PË>Jé½>fM±>0u¥>X>Ðî>0/>u!z>+i>(Y>lIJ>Ãh<>{v/>¡c#>þ!>¤>(Ý>^õ=®ä=ùºÔ=DÆ=ÕL¸="«=Á¦=\=I=³=ºo=qï^=xO=òA=®3=Î6'=M=¾Ï=äÂ=öÐú<Phé<×4Ù< !Ê<8¼<
¯<ãã¢<<6<ND<ÐNt<âXc<YS<KàD<A57<}*<§<I£<Mc	<õ²ÿ;\òí;DmÝ;ÇÎ;¿¿;o²;¦;Þ;Ê;ÞÎ;µt?¦1¿aÓ£=4x3?É¼y¿ÅW=>³Ðe?E¿EP¿&?úîp?Ëwp½ÉHs¿3¿½å>ÛÁp?ÊHZ?_)>>ÉV¿Yzw¿kj¿Å¿4O=î?\¾c?$ÿ?èh?$M+?pö±>\P;:£¾Ñ¿Ü|J¿Tl¿*)}¿1¿÷)u¿åa¿±H¿<Ý)¿êM	¿g·Ï¾6V¾¼>¾K½yÛ¢=À{:><>m»¸>Ááß>Äu?x?5è? %+?Çî5?k??×½G?O?ígU?ùZ?=Ô_?d?È¾g?mój?÷¼m?
)p?¡Cr?Bt? ­u?Iw?Ã>x?¯Gy?d-z?ôz?#¡{?»6|?_¸|?´(}?}?ZÞ}?f'~?®f~?~~?ùÌ~?ö~?·?8?IS?mj?x~?Ó?Û?à«?&·?éÀ?^É?°Ð?×?Ü?Gá?eå?öè?ì?¹î?
ñ?ó?Èô?Iö?÷?·ø?±ù?ú?Eû?çû?tü?íü?Wý?²ý?þ?Fþ?þ?´þ?áþ?ÿ?)ÿ?Eÿ?^ÿ?tÿ?K»v?µ|¿Yî4?ÕEC<¶dS¿»0_?ëD>øþ¿òt=û?>5Xl¿÷F,¿ëÉ>¤~?"J"?Êìl¾ðñb¿eXs¿iû¾¸3>îé:?±ð|?Rªk?3[?Ù->é?¾ëÓ#¿b¿÷í}¿hqz¿'û]¿q0¿äï¾»o¾Í8º2y\>^ÌÐ>ðq?íê4?IP?¼;d?7Rr?{?M? Ê?Ü@}?Sx?¤q?0li?¦P`?SV?îkL?½B?Ì7?¥-?6¦#?øû?O«?S½?koþ>æ:î>àÞÞ>äYÐ>¨Â>«Ãµ>¹¥©>UF>>¡>lI>o>ÚÇ^>yO>òNA>_4>¦'>ú>ìW>]P>	ðû= ê=¸NÚ=â3Ë=ð"½=ì	°=Ø£=}=Àë=ò=Øu=ÇËd=íT=­'F=:h8=¦+=´=ä=jN
=µ =Mï<ÅëÞ<rÏ<.Á<¥³<-§<(<hÅ<¸<¼z<bTi<!Y<J<r<<¡ù.<ÎÓ"<ß<Ö <Ã6<W5ô;ö@ã;éyÓ;FËÄ;!·;¸jª;Ô;9;NT;l> -'>5¿Xû?)d¿|Æú¾k={?ø·;¶¿Æx?<ºj}?ý¾Ä>¹[=¿Êmk¿áÕ½/üE?´y?¸éì>0ü¾Nù^¿Ä|¿'í.¿4Ù¾úÇ>xK?H|?Q5v?Ñ¶D?Y6ð>¡ø>;3T¾Zÿ¾³Ö9¿?)b¿}ìx¿ôÿ¿öþy¿Å¾i¿ÏùQ¿ö5¿';¿ëèç¾ª¥¾êñG¾Ñ½xG%=F>Þx>Â©>¤<Ò>ºö>¤?n?ç&?;22?¦"<?ÙÜD?áL?4S?ØY?'^?¡b?Kyf?MØi?×Æl?0So?èq?vs?4!u?äv?xÕw?`ìx?9Þy?æ¯z?§e{?/|?³|?ÿ}?~h}?LÁ}?;~?àP~?~?¼~?îç~?r?ð-?J?tb?w?Ù?®?d§?D³?½?tÆ?+Î?ØÔ?¡Ú?£ß?úã?»ç?üê?Íí?=ð?Zò?.ô?Äõ?#÷?Sø?[ù??ú?û?¯û?Cü?Ãü?3ý?ý?æý?.þ?mþ?£þ?Ñþ?úþ?ÿ?<ÿ?Vÿ?mÿ?¼¿>?·ê¾þÐ¥½%È:?×}¿|å´>/??Î×Q¿%ö¾8_?è¤?ík)¿é°j¿û7<ÏCd?@ [? B©=öÔ6¿~Æ¿ë1¿"x½?3o?ê]z?æ<?{Ú¨>G>ü½9¿§P¿ôw¿4ã~¿`j¿Ï½A¿jq¿(¢¾uJ§½ï>±¯>³?'¦)?íIG?Û]?Ön?áx?¥G~?õÿ?0s~?~Kz?Þt?jl?×c?_Z?¦P?p¿E?^k;?×,1?¼'?éX?~ç?ûÕ
?-+?åÕó>¶-ä>f]Õ>×aÇ>è5º> Ó­>1¢> I>*>»>Òu>Æed>ÈT>õ3F>R8>²è+>Y >C>,Ã
>.>ð=âß=)eÐ=ÅøÁ=´=!	¨=d=È=±v=]$|=ÿ§j=laZ=U;K=Ø!==q0=ÛË#=n=êÙ==A°õ<¬¢ä<ôÃÔ<ÿÅ<¢@¸<Nv«<7<{<Ê,<#<àOo<¸²^<Í<O<¡Ù@<$v3< '<th<`<<Qxú;§é;æØ;×É;©Ó»;XÉ®;Ê¦¢;ê[;¿Ù;t¼*¿Æc?×(¿9/?·«>B}o¿c*?¥¢?bZ`¿¹ªú¾áûN?áê?? Ì¾ãû¿Éç¾²?Í??03?+½8¿=¿ðS¿Òk´¾¢U>îÇ,?­q?~?"Z?+?\±~>Ì¸¾½ÜñÍ¾0U'¿l	V¿?Òr¿þ$¿;l}¿§tp¿S	[¿k·?¿¯ ¿åwÿ¾å¼¾ <u¾¼í½*Û:vá=\úV>±.>íhÄ>4Zê>Y!?x6?!"?qb.?È8?íëA?iïI?WóP?1W?«o\?Äa?ð+e?D¶h?§Êk?xn?Ëp?¿Ðr?Àt?lv?iw?½x?	y?iz?¥({?NÎ{?á]|?KÚ|?F}?~£}?iô}?:~?;w~?Õ«~?dÙ~?Û ?	#?£@?GZ?{p?¶?^?Ê¢?H¯?º?wÃ?Ë?Ò?°Ø?õÝ?â?xæ?äé?Ûì?lï?¥ñ?ó?<õ?­ö?í÷?ù?òù?Âú?vû?ü?ü?ý?rý?Êý?þ?Xþ?þ?Âþ?ìþ?ÿ?1ÿ?Mÿ?eÿ?ûm"¾´³à>ÊO¿¥¤?¿K)©¾+4?º²¾Eí_¿Ñ¶?ýe?û¾.ü¿}¾¾¦Ó+?<ªz?3åÆ> ö¾%y¿Z¿¦A¾i¼>m&X?ô?Ä{X?.+ö>jíù< Ë¾Ïm:¿Í*n¿Ôï¿Ét¿zQ¿l¹¿Ë¾	%¾Ô*=Ý»>êì>ÕÂ?Fñ=?íÞV?ÿ$i?u?	Î|?cÐ?©N?ôù{?nv?3o?W»f?+e]?kS?{II?îõ>?y¬4?¼*?Òª ?²?Wç?¹?Fgù>Ãté>ZÚ>Ì>¤¾>ñü±>û¦>ó>>¶>½{>Új>+Z>ÄK>K)=>*0>$>¾>5>·d>ñ~ö=ótå=Õ=RÎÆ=×
¹=:¬=wK =²/=YØ=ú7=p=¯Õ_=èNP=fÛA=/h4=ã'===be=(N=+Ôû<Yê<VÚ<
òÊ<¤Û¼<¿¯<C£<Ç1<¡<ÿË<[Ku<åCd<kT<Ð«E<¥ò7<\-+<K<è;<Tñ	<¥] <Xèî;,RÞ;ËâÎ;½À;÷'³;À·¦;$;/_;DÂ|¿af?ù¿@2X½ËH?¹q¿f¡=Råo?"ø¾ú\¿÷Ûà>¸v?6å0<dm¿dÄ=¿ÜöO>÷äk?Ù`?ck>EF¿Xºt¿n¿+	¿?<=¤?x`?~á?á k?iq/?ª¿»>ìüµ<!{¾ª'¿tH¿%æj¿¢|¿Øl¿æþu¿x,c¿´I¿¢+¿é)¿´wÓ¾¾ãG!¾½=Ö4>(«>«i¶>ßÌÝ>ã ?=?ö,?Õ*?B]5?Më>?ÊMG?ó¥N?<U?K®Z?_?ÂÖc?Yg?nÈj?m?p?n'r?Çþs?âu?Üúv?È.x?Ó9y?`!z?êz?{?è.|?±|?Ô"}?ñ}?ñÙ}?#~?_c~? ~?}Ê~?òó~?Û?ô6?äQ?8i?l}?ë??2«?¶?gÀ?íÈ?NÐ?³Ö?<Ü?á?.å?Æè?ãë?î?ëð?ðò?°ô?4ö?÷?¨ø?¤ù?~ú?;û?ßû?lü?çü?Qý?­ý?ýý?Bþ?~þ?²þ?Þþ?ÿ?'ÿ?Dÿ?]ÿ?(¡j¿{?fx¿3I'?é¢=ç\¿tuW?í{>Ô¿3N<ÚÝ?q:>¬mh¿à2¿u¹>ý¨}?8à'?¯_R¾mò_¿u¿x¿ò>i£7?¿8|?=m?a?6Ý;>ÕÝ¾r!¿e·`¿³}¿*òz¿_¿1¿ló¾ÐUv¾ÿv¼+V>·Î>L?åþ3?-LO?¥¸c?7ûq?[áz?ú;?ýÒ?ï]}?Cx?Èq?®i?+`?¾ÛV?oºL?ákB?78?ìé-?xñ#?»D?<ñ?5 ?Óîþ>×³î>YQß>úÅÐ>çÃ>w#¶>²ÿ©>º>ì>ë>¨>o>¾@_>WúO>G¸A>¡j4>-(>gp>§>>Izü=hë=«ÆÚ=£Ë=õ½=Çj°=52¤=}Ñ=ç9=²]=`v=×Ie=fbU=âF=ßÍ8=?û+=# =Ôð=«
=ü =cð<±fß<òäÏ<£vÁ<­´<L§<ôç<@<Ù<ÕF{<Õi<JY<ý}J<&o<<"Z/<-#<qÙ<N<"<¼ô;M¾ã;îÓ;Ð7Å;·;¶Èª;Lí;ä;àÊÌ¾Ø?=>Îu>)ÈA¿T0?%Ë¿i@
¿§/x?©Úf=Ïú¿&6½¸{? Ö>è"7¿n¿"
¾cEA?z? ø>Ñ¾¾Ø:\¿!ß|¿¦\2¿+H/¾bÀ> I?¨§{?Rw?'¬F?tCõ>>NxJ¾wû¾KZ8¿a7a¿v{x¿ìý¿Vz¿tZj¿ÅR¿	6¿V6¿)êé¾ø§¾~ÄK¾=H½È=\}>@v>úA¨>÷Ñ>²õ>Ò.?²?Ð&?ùà1?9Û;?2D?LL?S?öâX?{^?Íyb?]f?2Ài?â±l?ú@o?zq?Khs?Iu?v?Ìw?äx?{×y?ªz?`{?Êþ{?å|?³þ|?£e}?Ò¾}?~?O~?ÿ~?8»~?¹æ~?f?-?LI?Ça?úv?V?=?§?ï²?C½?5Æ?ôÍ?©Ô?xÚ?ß?Ûã? ç?åê?¹í?,ð?Kò?!ô?¹õ?÷?Kø?Sù?8ú?ÿú?ªû??ü?Àü?/ý?ý?äý?,þ?kþ?¡þ?Ðþ?ùþ?ÿ?;ÿ?Uÿ?1ïT¿<<?¹ä¾C¿½Ô<?Lª|¿°>yª@?P¿ôCú¾ÃA^??ú%(¿êPk¿n³;Ó¤c?é§[?­²=y6¿ñÐ¿û2¿ñX½åh?²No?Îz?¼X=?Dª>¬÷½[¿@ÏO¿ßÖw¿fí~¿Új¿JüA¿Ð»¿§Å¢¾±©½|Ó>%4¯>M?×z)?$(G?Â]?«òm?|x?÷B~?ýÿ?w~?ORz?ù&t?åtl?Ô§c?$Z?ÞP?×ÌE?¿x;?	:1?,'?ee?ó?á
?*6?Åêó>~Aä>pÕ>wsÇ>~Fº>â­>!@¢>ÕV> >½>F4u>Ázd>¨ÛT>=FF>`ª8>ø+>." >>Ð
>:>jð=âöß=xÐ=Ö
Â=Q´=Ë¨=(s=[=U=ä;|=å½j=ÎuZ=MNK=3==à0='Û#=>|=)ç=ò=3Çõ<¸ä<Ô×Ô<Æ<ØQ¸<S«<<x<²9<&¡<;fo<Ç^<)PO<¥ë@<è3<0'<ùv<â«< <¸ú;m*é;NúØ;äéÉ;5å»;¬Ù®;ýµ¢;j;è?-¿e?à~¿¦Û,?o¾$>òep¿ì(?{?YU_¿æþ¾fèM?¶A?ñÉ¾ÿ¿I7ê¾*{??ãø3?µ1½Âg7¿y¿{T¿RÛµ¾¤ÖR>@J,?yq?ÿ~?Ì_Z?Í?>>ÄH»½5Í¾¾'¿ØU¿õ·r¿¦¿Xv}¿p¿u)[¿¶Ý?¿{Ù ¿¥Îÿ¾Û¼¾Zäu¾LDî½q:ªñß=6{V>ýô>ì4Ä>+ê>?$?Ñ"?T.?ð»8?ÑàA?®åI?ÔêP?ÂW?.i\?a?'e?û±h?íÆk?Ëtn?¼Èp?NÎr?¢t?v?çgw?[x?Öy?yhz?¾'{?Í{?3]|?´Ù|?E}?£}?ô}?.:~?òv~?«~?-Ù~?« ?ß"?@?(Z?`p??I?¹¢?9¯?º?lÃ?Ë?Ò?©Ø?ïÝ?â?tæ?àé?×ì?iï?¢ñ?ó?:õ?¬ö?ì÷?ù?ñù?Áú?uû?ü?ü?ý?rý?Êý?þ?Wþ?þ?Áþ?ìþ?ÿ?1ÿ?Mÿ?¡<XÕ½,òÈ>@uH¿ý?*#'¿H¾æR~?wÂ¾ßþ[¿ù?'èb?Ð¾û¿H8´¾x/?§y?¯¾>{6ý¾  z¿b°X¿?á¾IOÂ>J®Y?bã?$W?Ðò>dÈ´<ÈÁÎ¾í´;¿çÌn¿»÷¿2s¿@«P¿³¿óQÉ¾!¾d=>¬¦í>m?x>?ëDW?mi?²u?»å|?Õ?âD?Gä{?¼Ov?Úo?¥f?$7]?]OS?sI?ÁÄ>?Í{4?\*?| ?î?x¼?Çï?`ù>ã*é>½Ú>ÔË>ñe¾>Â±>2ã¥>¬¿>lO>>Êz>)³i>­»Y>)ÓJ>Hé<>[î/>hÓ#>1>2>¤7>ö*ö=¹&å=<MÕ=yÆ=ªË¸=9ÿ«=³ =¶ü=â¨=Ì=×1p=!_=§P=A=u*4= ª'=¡=¡3=Ú=û}û<V	ê<±ÊÙ<¬Ê<ÿ¼<W¯<GT£<®ý<p<á<c÷t<Áõc<T"T<$hE<¬³7<Âò*<<)	<Â	<³1 <î;Þ;÷Î;ÔCÀ;¡ê²;­~¦;~ï;»õ?º~¿¢uk?9¿MD¼øèA?¹®t¿úé=¾Íl?vé¿ÈEX¿2í>´òt?ìI¼åo¿÷I:¿ïb>Im?T^?Kf\>»P¿¯u¿Ùl¿Ó¹¿ò<í¾
?a?
ð?Ê2j?B.?Û¸>Õ <¼Z¾!M¿jßH¿-ak¿HÐ|¿>Z¿¹u¿Àb¿I¿ä+¿ï
¿É8Ò¾%Ê¾8ñ¾ùL½i¦=D·6>ß>7/·>/~Þ>× ?e?Gk?Ù¶*?²5?Ù??sG?ÆN?Ã.U?1ÇZ?½¨_?«éc?Îg?¿Öj?	¤m?ap?Ò0r?ît?õu?þ w?4x?q>y?`%z?íz?{?1|?Ù³|?É$}?¢}?iÛ}?Ú$~?yd~?~?QË~?ªô~?y?~7?[R?i?Å}?8?U?l«?Á¶?À?É?oÐ?ÏÖ?TÜ?á?@å?Öè?ñë?¡î?õð?ùò?¸ô?;ö?÷?­ø?¨ù?ú?>û?áû?oü?éü?Sý?¯ý?þý?Cþ?þ?³þ?ßþ?ÿ?'ÿ?Dÿ?ÐÔY?0!\¿st?b¯}¿ÇÑ9?þÿ]¼éO¿¼æa?}â/>ú¿ É=åå?h{>¬ºm¿ý¼)¿ìéÎ>Kä~?c' ?¡Þv¾v	d¿ð¥r¿Ýñ÷¾øD;>!<?ä0}?k?Ñ5?Ã(>¥¾·$¿b¿~¿j?z¿ô]¿Ûz/¿¿î¾Þ^l¾ø±:Ä×^>0ÔÑ>/à?fC5?EP?Àld?¬rr?Ï${?æS?KÇ?Ä5}?æAx?ìxq?"Si?L5`?rV?YNL?íþA?ô®7?-?á#?à?ý?(¤?y?þ>dî>Í³Þ>91Ð>½Â> µ>à©>&>I>0>a/>îén>`^>_O>S'A>iã3>'>Øú>ù9>u4>	¼û=/Vê=!Ú=Ý	Ë=Ïû¼=~å¯=¶£=ö]=[Î=ù=­¥u=]d=îÀT=þE=ÿA8=y+=ü==¼1
=^ =Zï<½Þ<GÏ<#äÀ<X³<m
§<ãq<`§<<z<ú#i<~ôX<¢äI<pà;<UÕ.<²"<of<ã<<­ô;Ñã;
NÓ;r¢Ä;û¶;]Gª;ît;{?¹¯¿ê>´e	>£0¿üù?D\¿¿Þð¾2|?dí\¼º:¿2ç<÷ñ}?Åô½>3£?¿í)j¿[¾½w·G?ùrx?k®è>ý0£¾Åú_¿¿®{¿V-¿QG¾UÒÊ>FWL?ê|?3Ýu?ZøC?ôMî>ùû=ÛÚW¾þD ¿ëd:¿Ob¿"y¿ñÿ¿VÝy¿®i¿À¬Q¿kÆ4¿tÜ¿'ç¾©`¤¾¤F¾ìR½âZ*=Ä>§ïy>Xª>Æ«Ò>í÷>¾Ð?¤A?­	'?ÂP2?|=<?côD?}L?FS?Y?»4^?¥b?´f?[ái?¶Îl?Zo?Ùq?/{s?®%u?Æv?×Øw?Lïx?Âày?²z?g{?Õ|?!|?<}?i}?:Â}?	~?Q~?6~?#½~?bè~?Ö?G.?`J?¶b?Éw?
?Ø?§?d³?¨½?Æ??Î?êÔ?°Ú?±ß?ä?Åç?ë?Ôí?Dð?`ò?3ô?Èõ?'÷?Vø?]ù?Aú?û?±û?Eü?Åü?4ý?ý?çý?/þ?mþ?£þ?Òþ?úþ?ÿ?<ÿ?Ûf?7Nl¿9JX?Øå¿]=SÒ!?é¿Sîê>¤©+?r_¿IÉË¾©;i?z?/V7¿eûb¿¯ê=oÚj?,2S?"«ø<wX?¿®¿þ)¿¯¼º?r?x?iº7?è>¦ÿ¾²1¿±WS¿6y¿b~¿ßh¿zñ>¿ 	¿-¾½>=µ>ßÛ?<+?ßÆH?bô^?~Än?å y?	{~?þ?qF~?ýy?£µs?Öîk?-c?ïzY?2nO?÷&E?=Ó:?á0?R&?Ë?í^?/S
?r®?Üèò>+Mã>;Ô>ÍÆ>¦y¹>"­>¡>­>>tø>t>¹wc>ÈéS>}dE>Â×7>-4+>
k>an>ø0
>P¦ >Aï=¨õÞ= Ï=À+Á=Ë³=dW§=¿=½ó=Lç=g{=¯i=#zY=dJ=}Y<=÷G/=O#=~Ì=C=ºu=à«ô<[°ã<sâÓ<C-Å<V}·<Àª<æ<5Þ<S<×<2Rn<§Æ]<aN<3@<æ·2<O&<µÃ<<b<Ìnù;è; Ø;É;»;®;]ú¡;mFÝ¾öîÄ>ò¿ÌN?¿'^F?wrØ<§uc¿dê=?"Ýù>3Ùj¿¦Ó¾ðrZ?¬2?xÅì¾ÍA¿ÃË¾®?Íá?9*?Ùx²½i?¿ñð¿·0N¿¤¾f8u>ôD2?æs?þ)}?âÙV?6w?è@j>9°å½hAÖ¾Ñ*¿-X¿õs¿¡f¿ô|¿lo¿4Y¿?>¿Ò¿Úû¾\¬¸¾ÐÃm¾ºÌÞ½»´ù;~Ví=)\>Ö¼>W¶Æ>/jì>ª?Y?äL#?b/?ìW9?»iB?¡]J?¿SQ?giW?#¹\?ÌZa?²ce?Èæh?Øôk?³n?fëp?gìr?Â©t?@,v?{w?ix?y?Kuz?Û2{?(×{?e|?ðà|?ÚK}?{¨}?»ø}?A>~?zz~?¤®~?ÓÛ~?ö?Ü$?8B?¥[?«q?½?A?£?ò¯?­º?÷Ã?Ì?ûÒ?Ù?=Þ?Ãâ?®æ?ê?í?ï?Ãñ?¬ó?Sõ?Áö?þ÷?ù?ÿù?Íú?û?ü? ü?ý?xý?Ïý?þ?[þ?þ?Äþ?ïþ?ÿ?3ÿ?èý=CÉx¾]G>úÂ2>v"¿Øt{?^»G¿õM¾4|v?¿òG¿²|#?^2S?æ$Ê¾ÈJ~¿Úi¾D??´t?4\>¯K¿A!}¿ÄO¿9?f¾§Û>G-`?A?ØåP?¥ß>¿¼ß¾JA¿"~q¿øú¿¹tq¿ÄùL¿u¿OK¿¾ø¾ ·=¥ý>ÕÔô>Õb!?ÈÎ@?ÔY?£©j?Õzv?NK}?;é??À{?Ãu?Rcn?^Ðe?>h\?¦wR?<H?fè=?Õ¡3?§)?K­?'&?Ãü?»8?%½÷>¶àç>ÜØ>®Ê>P½>Ü½°>sî¤>4Ú>x>xÀ>hRy>°Sh>nsX>Â I>bË;>°ã.>ÂÚ">g¢>-->\n>í³ô=cÉã=áÔ=|[Å=±·=øª=# =	=ñÔ=F=Ân=D3^=vÉN=îp@=Ô3=©&=â=pU=Q=ýù<(£è<]}Ø<`vÉ<Rz»<²v®<GZ¢<<
<hÕ<hs<Îb<ÝR<õ9D<w6<í)<û <&<9ï<ëÚþ;Q)í;/²Ü;¯_Í;¿;½Ø±;Í¥;³	~¿aTx?N{}¿¨|?'ØE¿Ä@>" ?;ë}¿XJ>¸à[?6Þ¿UÿD¿Á­?ö3k?,ì½ (w¿¢Ý)¿¬h>?t?FÏT?Âç>¼¿Öqy¿eAg¿;÷¾ñ1=ú?RKf?þ÷?mf?ÑÚ'?¶è©>ÒùJ¼1ª¾]¿­oL¿÷um¿}¿´ù~¿Äut¿Õ`¿æ¸F¿dh(¿ñÆ¿ ¤Ì¾"T¾=}¾Ûí×¼ ç¬=Ã?>=U>ßº>iá>¿7?Ü¼?×?Ø«+?^e6?FÓ??"H?æWO?¼­U?6[?Z	`?Ò=d?çg?rk?hÛm?{Cp?Zr?1+t?m¾u?Kw?ÈKx?ùRy?-7z?ýz?}¨{?=|?ä½|?}-}?-}?ñá}?*~?`i~?Ô~?þÎ~?Ù÷~?<?á9?mT?jk?R???n¬?¡·?TÁ?ºÉ? Ñ?M×?ÁÜ?zá?å?é?.ì?Öî?#ñ?!ó?Ûô?Yö?¤÷?Ãø?»ù?ú?Mû?îû?zü?óü?[ý?¶ý?þ?Iþ?þ?¶þ?ãþ?	ÿ?*ÿ?V¬D¿ÿ!?Öß)¿^ËT?Ì¹}¿_?Zûx¾T+¿eu?¢§k¼S{¿þôr>Ðé{?G\¼&x¿F¿Ì3?9û?üß?B¨¾Õm¿_wk¿/oÖ¾>kÆF?Ìû~?e?m?wð=M¤¾Y©,¿ìüf¿¥¿¯Px¿ÎxY¿R
*¿YVâ¾"tS¾.É<F_t>Æ+Û>QÅ?ªa8?3§R?÷#f?s?Å{?·?P¤?·Ì|?S¢w?-°p?zlh?:_?ajU?w?K?0î@?0 6?±u,?""?æ?Ç ?¾?ü>Rnì>+Ý>i¾Î>b$Á>W´>QO¨>0>'p>g>
~>?.m>ñû\>ÜM>D¾?>2>ÿI&>Ö>)>(6>1âù=Êè=~Ø=É=L»=®=¢=>>=Â=A =Õs=Rìb=Í.S=SD=§å6=Ý4*=Ae=Cg=e,	=PNÿ<ðí<BÝ<y¿Í<Kw¿<Ñ,²<vÎ¥<ÚK<À<ø<®x<ôjg<ZW<¶fH<}:<-<@~!<H<Ù<#<5ò;Adá;L¾Ñ;u.Ã;m¡µ;<©;à#¿¨9F?P?¿Q?°0¾ù¾=Qx?ðp>¿_×¾8ù?uá>¾§°x¿û<6>ú?|ñ|>´ÉR¿]¿CÛE<hV?¥Îq?>Á>êÈ¾Rwh¿Ãæw¿.R!¿fQ¶½*_ä>o÷S?r:~?éur?=?)»Ü>«2³=>y¾¿^?¿® e¿{z¿<ì¿wx¿ì[g¿FäN¿ò1¿x¿~>à¾Ö¾na9¾w½ÿX=i!> Ò>bj®>*Ö>2¥ú>£b?á¥?hD(?f3?+1=?ÊE?RM?¦éS?e«Y?H±^?c?+âf?3j?(m?o?ÄÅq?þ©s?NNu?»v?i÷w?Ï	y?¾÷y?Æz?Ôx{?Í|?|?y}?Ns}?ªÊ}?X~?èW~?²~?ãÁ~?ì~?g?]1?M?e?Êy?Æ?Y?Ö¨?´?¢¾?eÇ?ûÎ?Õ?=Û?*à?oä?!è?Të?î?ð?ò?`ô?ïõ?H÷?sø?vù?Wú?û?Áû?Sü?Ñü?>ý?ý?ïý?6þ?sþ?¨þ?Öþ?þþ?!ÿ?")t¿Ø½?äñ¿z?BßP¿1¶>)É>±üx¿*º(?Bvù>Ot¿Y¾~x?]l½>© Q¿Â9O¿àDX>Íßu?b¥@?.£½»O¿º¾{¿û]¿é$z= '?¿w?3ís?/Y,?!g~>J	N¾Ö»¿ÑY¿{¿}¿Vd¿§ä8¿Ú¿ff¾5W%½áÞ7>UÑÀ>!¾?£/?ÚØK?5a?bLp?ýèy?«Ü~?)ñ?¥à}?Qy?Õr?!çj?ða?FX?Z/N?[äC?¿9?DZ/?nX%?û?"??«?	?Í§ ?àõð>tá>ÌÊÒ>÷ôÄ>ªí·>®«>z. >pf>;M>òÙ>]r>Ka>R>e°C>ê@6>¿¸)>H	>¥$>²ý>
ÿ=Üoí=ÖÝ=OºÍ=ã|¿=i:²=Þá¥=[c=°=ñ¹=`èx=M¥g=W=ªH=o´:=À-=±!=y=µ=¾O=²ò<"³á<Ò<AtÃ<îâµ<£B©<ª<t<f<ÐÜ}<=l<Ö[<vL<_><(1<Û$<i<åÂ<Ù<Ñ@÷;Ræ;êÖ;j?Ç;j¹;«¬;Dç>ÿ7½ìú©<ÁÓ[¾Ø?I>o¿ìik?8
n¾<@¿Ú_?H>{.z¿
+v¾àÕm?b§?¾O¿±9z¿¾½(?á/??ì9¾	¸M¿­¿TðA¿bö¾a>$K=?Dùw?´Ãz?º¤O?Ú?ý>>w¤¾ù~ç¾1¿ú\¿Õ/v¿Ê¿ÕÖ{¿'m¿¥~V¿¶[:¿ÈÜ¿ëró¾o°¾X^¾äÀ½P®<y>1lh>æ¢>Ë>¨»ð>(ù?^¼?XÏ$?Z0?:?´pC?ûCK?4R?ZX?¡R]?àa?,Øe?!Li?÷Ll?Hén?ì-q?)&s?ãÛt?ÀWv?M¡w?"¿x?ÿ¶y?äz?-H{?£é{?u|?Ðî|?àW}?æ²}?Â~?F~??~?´~?èà~?]?¬(?E?^?$t?â??+¥?W±?â»?Å?êÌ?ÃÓ?±Ù?ÓÞ?Fã?ç?uê?Xí?Øï?ò?ãó?õ?êö?"ø?0ù?ú?äú?û?+ü?¯ü?!ý?ý?Øý?"þ?bþ?þ?Êþ?óþ?ÿ?ÿU¾?3È!¿N?CÚO¾ßÙ§¾v^?]´n¿a >Â[?Õ4¿ÉR¿°I?hw2?í¿zu¿âoÓ½å V?g?f2>ÔÅ&¿sÐ¿G?¿i»¾)?l½i?åô|?tE?À> ½vù¾Ì"J¿|u¿ê¿[~m¿ÉF¿d5¿Æc®¾¤I×½Leõ=ÖÚ¥> U ?=&?[D?à[?o¦l?·w?Iå}?ü?K¼~?èÐz?Ñt?ø?m?íd?l
[?øQ?«ÊF?Kv<?142?	!(?TR?º×?'¼?¿?Dwõ>.¹å>$ÓÖ>FÂÈ>»>¸
¯>	V£>T[>î>öp>ßv>v	f>ùOV>À¡G>î9>þ&-><!>å>øÄ>¼>Bò=èá=déÑ=LbÃ=!Ûµ=B©=`=n=s=$û}=3^l=Kù[=õ¶L=,>=IK1=èý$=Ø=ÿâ=Qø=n{÷<þMæ<¡QÖ<3qÇ<¹<Ï¶¬<y¹ <(</<<<q<S`<6ÀP<%BB<¡Å4<É8(<û<»¬<¢<Lü;dÈê;{Ú;^PË;Ì2½;°;w?øöS¿ifF?oZ¿·«z?,Úq¿Zý>°ÿ¸>7×|¿ªN?Î45?¼+I¿fÜ¿g7?T?I¾Í¡~¿¿mÜ>¿|?E9B?ó =¹S*¿×}¿p\¿iÒÐ¾3m>m½"?âPm?26?Z_?á?,<>¯s½¢!¿¾R!¿ü#R¿_±p¿å~¿Ö'~¿@2r¿®]¿]®B¿ð#¿D¿pEÃ¾¼/¾ë¾Ôr)¼MË=M>yª>VÀ>´æ>ã?¬Ä?ñL!?ÖA-?«Ê7?A?,I?zHP?ïV?oí[?@©`?Éd?>`h?Ùk?7n?p?µr?/gt?|òu?uIw?órx?ðty?Tz?{?¾{?GP|?Î|?â;}?¥}?¿ì}?ß3~?{q~?Ú¦~?Õ~?ý~?Ì?Õ=?ÙW?an?ä?Ê?l¡?®?¹?Â?ÏÊ?ðÑ?Ø?uÝ?â?æ?é?ì?.ï?oñ?có?õ?ö?Ï÷?èø?Ûù?®ú?eû?ü?ü?ý?iý?Âý?þ?Qþ?þ?½þ?èþ?ÿ?$+?w¨¾-99>;C¾ªÒ?ôµ[¿È}?¶ß¿µÅÉ¾Yø?h¾íjf¿î>ïk?"e¾§¿lÐ¾1î$?¢A|?FiÕ>ãôè¾zw¿¦Ô]¿~¡¾Q;²>ôXU?ÿÿ?¢ÔZ?¶iý>¤³:=tÄ¾8¿1m¿+Û¿uÞt¿éR¿!¿×Ï¾­-¾ýNt=^>Û#é>?þ<?'V?5£h?1u?Ù¢|?MÆ?z_? |?H¥v?§vo?Gg? ·]?ÕS?ã I?¡M??L5?Íà*?xý ?sj?Þ3?Ña?^òù>¬øé>\×Ú>>Ì>á¿>+e²>Õ{¦>ËN>zÕ>>(µ{>jj>sZ>QK>¦=>º0>n$>Ï>ú>;µ>÷÷=± æ=?Ö=GÇ=´{¹=!£¬=M­ =Æ="-=ç=q=q^`=2ÎP=ßQB=rÖ4=2J(==E¾=á =$nü<Õèê<¯Ú<#nË< O½<ø*°<Fð£<Ú<¢÷<<^áu<Ïd<ôìT<³$F<$c8<+<t¬<<±E
<'¬ <uzï;$ÚÞ;RaÏ;|ûÀ;³;Ñý=?¿¾q¿øÆ{?zw¿ßØZ?²c¿"¾ñ¾S?dHk¿Óaz< Ãt?Dß¾tb¿Ë«Ê>7y?QU=yÉi¿dÊC¿{.>³h?x÷c?+>7ÿ¾¿êr¿ÿüo¿í|¿¡Ôº9¦Ù?¾q^?â»?Ëgl?öÜ1?LÁ>N	=U¾¿F¿ûj¿xL|¿R¿nyv¿wëc¿ÛJ¿3­,¿eC¿y°Õ¾30¾Ót%¾ßÖ+½A=y{1>*$>uµ>ËÜ> ôÿ>*¿?|½?i*?5?.>?G?kN?1ßT?½Z?/l_?ë´c?åog?Ð®j?Sm?8õo?¡r?2ðs?:u?áïv?A%x?1y?7z?çãz?º{??*|?­|?T}?è}?P×}?N!~?fa~?ë~?É~?ªò~?¿?þ5?Q?h?Ì|?a??Êª?5¶?À?©È?Ð?Ö?Ü?áà?å?©è?Êë?î?Øð?àò?£ô?(ö?z÷?ø?ù?wú?5û?Úû?hü?ãü?Ný?«ý?ûý?@þ?|þ?°þ?Ýþ?ÿ? |?0t¿"É]?dµ_¿5Zv?¢Æ|¿°Ê5?~Jõ;ÃËR¿«_?õ@>Ôÿ¿áË=ù?Bb>Ìl¿eÖ+¿Ê>C§~?Oë!?Y¨n¾ó"c¿«9s¿Nëú¾¬5>B ;?ü|?ak?9(?Pî,>¯ª¾û#¿*b¿¡ó}¿Ähz¿Øç]¿D÷/¿K©ï¾Ln¾<DºËâ\>NúÐ>#?Vú4?P?GDd?àWr?¼{?¢N?Ê?ï>}?Px?Ýq?Õgi?äK`?PV?ÉfL?B?kÇ7?-?H¡#?4÷?º¦?ò¸?gþ>û2î>a×Þ>ÐRÐ>c¡Â>e½µ>Õ©>Ï@>Û>0>äD>$o>ï¿^>O>HA>ð4> '>c>µR>K>üæû=1~ê=ÞFÚ=,Ë= ½=°=!Ò£=x=¢æ=/=ÁÏu=Ãd=båT= F=a8=t+=V®==lI
=j° =§ï<¹ãÞ<kÏ<6Á<³<'§<<.À<°³<³z<õKi<²Y<AJ<§ <<Pó.<íÍ"<f<¿û<2<,ô;Á8ã;FrÓ;+ÄÄ;ø·;ç&¾¸¾1¶ÿ>èëø¾Ô;>">]>4¿*þ?C¿Vù¾òi{?;F~¿þ¡n<_}?-Ã>çÁ=¿6k¿ÉÑ½ÂIF?;óx?-ì>Ö·¾b&_¿µ÷{¿³.¿´¾VyÈ>sK?µR|?&v?½D?káï>G>"ÖT¾Kÿ¾}ï9¿õ8b¿Åóx¿ûÿ¿"ùy¿´i¿jìQ¿5¿®*¿FÇç¾mþ¤¾Ò±G¾½¥)&=^Í>y>E ©>ýOÒ>ÙËö>7¬?A!?í&?72?R'<?óàD?xL?.7S?Y?f)^?µb?{f?áÙi?6Èl?aTo?ñq?ìvs?ü!u?v?Öw?âìx?ªÞy?H°z?üe{?x|?ó|?6}?®h}?vÁ}?_~?ÿP~?¶~?´¼~?è~??ÿ-?"J?b?w?á?µ?j§?I³?½?xÆ?.Î?ÛÔ?¤Ú?¦ß?üã?½ç?ýê?Îí??ð?[ò?/ô?Åõ?$÷?Tø?[ù??ú?û?¯û?Cü?Äü?3ý?ý?æý?.þ?mþ?£þ?Òþ?úþ?´¶Ê>O¿.q?ñ´t¿¸c?©'¿_0>?ÑÈ¿:?åG ?çe¿äÅ³¾ñm?÷>ow>¿T^¿ßåã=µn?¿¬N?ÉYú:ÊÇC¿.f~¿F%¿²Ò×¸»Ê?\ûs?æuw?MÚ4?nC>T&¾Ïó¿{U¿Ùy¿©~¿õug¿Çf=¿P¿´^¾ nz½V$>I>¸>¹ ?	,?éI?`_?w+o?g>y?~?íû?#-~?Ñy?O|s?I«k?rÈb?+Y?"O?ßÓD?b:?=E0?¥=&?ã}?¦?
?¢j?hò>Óâ>ìÔ>-Æ>Z¹> Â¬>W1¡>	Y>]0>l®>s>höb>qS>ËóD>n7>Ò*>>*>á	>S\ >gûî=AuÞ=lÏ=f¼À=éc³=Ûö¦=:e= =i=hz=(i=üX=#ïI=ªì;=¯â.=À"=Ät=ôñ=¿)=tô<¿,ã<ùgÓ<H»Ä<D·<Û]ª<:<¹<ÆJ<<jÈm<oF]<ÍéM<*?<P2<fï%<;j<Í±<å·<Þø;^ç;:×;ÚÈ;f º;k¿ê?C3¬¾Ød>8ìé¾ÕtA?|}¿vQ?t(½H¤[¿aG?ý5á>`³o¿åã¼¾Î`?+?ÈÑý¾ùh~¿½9¼¾g?áÿ?Gñ$?­ä½ABC¿  ¿9#K¿¾H#>/5?u?|?í	U?zë?6_>vÑú½·½Ú¾Ð:,¿	QY¿jt¿e¿®|¿ÀÙn¿ÎX¿Â=¿Î¿¸ù¾U¶¾Ë³i¾×½e±6<ô=ª_>ª>÷õÇ>ní>6?x?±#?«]/?¥9?á­B?UJ?öQ?W?íà\?y}a?äe?i?°l?°n?¥üp?aûr?Á¶t?7v?Zw?å¦x?û¡y?¬{z?b8{?óÛ{?³i|?ä|?øN}?/«}?û}?H@~?;|~?)°~?$Ý~??Ù%?C?c\?Or?K?½?ú£?O°?ýº?=Ä??Ì?/Ó?0Ù?dÞ?åâ?Ìæ?-ê?í?¢ï?Ôñ?ºó?_õ?Ìö?ø?ù?ú?Óú?û?ü?¤ü?ý?{ý?Ñý?þ?]þ?þ?Æþ?ðþ?Í¿&o¼<µ>¯læ¾áÖ­>¼2¼ù¾p?ùþ]¿û<ùk?~f¿«4¿¼7?¤C?Éõ¾kÅz¿:¾IÀK?ðn?ºur>Øí¿ûá~¿Þ¡G¿Ç=5¾gð>p4e?½I~?IVK?øÇÏ>¶ìA½ì¾åÓE¿ís¿Õ¿~o¿ÂÅI¿Õ"¿gÊ¶¾pø½ÏÖ=#>_Óú>Ù#?÷¾B?[zZ?k­k?ew?m}?õ?ê~?){?³Ju?]Ñm?,e?q¸[?¸ÀQ?dG?.=?¢é2?pÒ(?gþ?}?[?k?¨ö>Êæ>Õ×>òµÉ>g¼>Nâ¯>^ ¤> >Ã>)>Êx>Ø+g>%_W>ÝH>ÂÚ:>A.>~	">Uß>aw>øÄ>Pxó=g£â=öÒ=\Ä=Ä¶={ª=UR=qY="=ù@=}m=]=´½M=¸w?=ã.2=¼Ñ%=üO=x=£=<¹ø<Áuç<ßd×<XqÈ<gº<£­<è¡<CQ<Ûá<Þ+<ÞDr<+sa<YÌQ<¬;C<Õ­5<Þ)<T<Úg<Ä=<¦ý;úõë;-Û;UÌ;Õ%¾;¦MT¿ù?ælo¿ðd?ÙÊp¿V÷?O_¿®Ô±>ûþ>l÷¿µÐÊ>ÔäI?é5¿¼B2¿%?fÁ`?åçM¾¼{¿ý¿s.¼>k¸x?âËK?!¿=NB ¿Rõ{¿b¿ä¾
tì=?Áøi?¶?íc?;{"?>ë*½uÄ´¾P¿NZO¿$!o¿~¿i~¿Ws¿-_¿²D¿ )&¿èm¿êëÇ¾W¸¾G®¾&¶¼q@¼=·!F>L>½>^+ä>_?©Ã?|i ?%y,?7?óq@?5¤H?ÑO?V?Ì[?>Z`?Ed?\$h?ÅKk?Á	n?Àkp?}r?It?ÄØu?%3w?_x?(dy?Fz?í	{?®³{?ÎF|?LÆ|?Å4}?|}?iç}??/~?zm~?a£~?Ò~?ú~??á;?(V?êl??±?y ?F­?\¸?öÁ?FÊ?zÑ?¶×?Ý?Éá?Öå?Xé?aì?ï?Iñ?Bó?÷ô?rö?º÷?Öø?Ìù? ú?Yû?øû?ü?úü?bý?¼ý?
þ?Mþ?þ?ºþ?æþ?ôï¿K?ÚS÷¾Fª>LÉ¾H!?j¿yx?s¬ï¾\¨ø¾0?~T¾?Ín¿2Ë>'´q?¥"¾WX~¿|ë¾Æ>?~?R²ê>¤ Õ¾ß§t¿âob¿T²¾9§¢>Ô Q?ç?ù5^?8?d=Ëº¾J4¿¹6k¿¼¬¿v¿gU¿:$¿ØpÕ¾ë`9¾uG=T>Íä>àÄ?a;?8U?íÞg?W³t?R`|?rµ?w?ÄW|?Èõv?ÌÙo?Ûwg?Ñ0^?RTT?í"J?.Ð??5?_+?¦x!?Xá?û¥?ÕÎ?×Áú>u½ê>Û>å;Í>O¸¿>µ ³>Ý§>º×>¿U>>¬|>:`k>iL[>@IL>XF>>ò31>%>7¥>ø>m->ëô÷=NÑæ=ÚÖ=wüÇ=0$º=@­=Z?¡=¿=µ«=ºü=]òq=*a=:Q=½C={5=fã(=/+=øB=_=ÿSý<¿¾ë<ÁaÛ<f'Ì<û½<iË°<¤<Ë<ïx<ì<QÁv<æe<ä®U<-ÙF<9<V2,<ä= <è<£Ã
<[!<Tð;!¥ß;7Ð;C«Á;,Cµ<Sè?n&`¿°mq?qk¿;ÎF?ö-Í¾mì|¾8b?+È_¿)£½pmz?¸¾Ôùj¿b´¨>Á|?^~è=kc¿PL¿Jø=#c?/Äh?Ï¹>wÙî¾Løo¿½r¿mÔ¿Ù÷á¼";þ>;P[?h?fin?bk5?	Ê>[iN=Ú¾ÿ¿0aD¿Ï±h¿$Å{¿@²¿à*w¿5e¿ëK¿9.¿Üå¿AÿØ¾Bo¾E¯+¾çQC½½W=ýw,>ÖÛ>qø²>­µÚ>)Mþ>å?¶?')?î4?C->?,§F?N?ÉT??Z?	2_?Dc?ÎCg?wj?ü_m?AØo?zýq?[Ús?Gxu?pßv?ÿx?3%y?~z?Úz?¬{?D#|?§|?}?]}}?aÓ}?å~?r^~?\~?ËÆ~?¿ð~??4?ÏO?kg?Ý{??ç?.ª?¯µ?¤¿?DÈ?¼Ï?4Ö?ÎÛ?¨à?Üä?è?¥ë?`î?½ð?Èò?ô?ö?k÷?ø?ù?mú?,û?Òû?aü?Þü?Iý?¦ý?÷ý?=þ?yþ?®þ?Ûþ?^¿	Tv?¹Üz¿§ i?Ïj¿LS{?2×x¿5(?ÝÖ=ÿ[¿4ýW?®w>Å£¿m~<®ã?t½7>²h¿ws2¿C}º>¨º}?t'?ÝT¾Ý%`¿~ÿt¿;5¿në >ìÚ7?gE|?#m?.?ò:>9I¾±!¿vÎ`¿×}¿çéz¿_¿1¿ò0ó¾UÜu¾f_ö»V>ß;Î>d_?k4?XO?HÁc?ô r?äz?'=?vÒ?\}?P}x?TÄq?4ªi?k`?½ÖV?JµL?±fB?8?Ûä-?ì#?õ??¦ì?Òû?xæþ>è«î>ÖIß>ã¾Ð>9Ã>.¶>Ëù©>1>ëæ>2æ>>o>Ð8_>ðòO>^±A>/d4>*ü'>Ìj>Q¢>±>7qü=õþê=Í¾Ú=BË="½=ld°=J,¤=ûË=Æ4=ìX=+Wv=Ae=³ZU=¸F=3Ç8=
õ+=] =të=«
=^÷ =¸ð<¡^ß<pÝÏ<¦oÁ<-´<?§<Sâ<<ùý<Ã={< Ìi<nY<­vJ<Xh<<ÍS/<¸'#<õÓ<I<bz<2³ô;¶ã;ææÓ;±0Å;¢lZ?g¾³L¾ÛÓ>£Ï¾eËB>azp>%û@¿ØF?Ã³¿ôk	¿Åhx?:Y=ð÷¿	Íð¼[Ø{?oÕ>=7¿jn¿¾ÐA?|rz?çU÷><|¾»i\¿¸Ñ|¿l#2¿¿#.¾âÀ><HI?²{?À	w?|F?Úîô>hÜ>ãK¾"Gû¾Zs8¿\Ga¿ûx¿&þ¿íPz¿MPj¿@¸R¿.ú5¿á%¿Èé¾µù¦¾PK¾·Î½A«=±>ì7v>W¨>`(Ñ>yÃõ>6?í¸?Þ&?Oæ1?êß;?O¢D?OL?:S?¶åX?á^?ä{b?h_f?ÈÁi?B³l?,Bo?!{q?2is?u?;v?Íw?åx?ì×y?pªz?ë`{?ÿ{?%|?êþ|?Óe}?ü¾}?:~?$O~?~?P»~?Íæ~?x?-?YI?Òa?w?_?D?	§?õ²?H½?9Æ?÷Í?¬Ô?zÚ?ß?Ýã?¢ç?æê?ºí?-ð?Lò?"ô?¹õ?÷?Kø?Tù?9ú?ÿú?«û??ü?Àü?0ý?ý?äý?,þ?kþ?¡þ?Ðþ?ýPß>¶>hI¿ ¿m?5ûq¿ßÝ_?OÖ!¿(+æ=Ê_?qö¿&ý>;c$?°c¿_[¼¾DTl?iþ>Xø;¿ `¿Ò\Ê=wõl?MP?dÓD<8B¿¶£~¿Ø$'¿Â£û»[?Ñws?4Ûw?á5?5Ú>
!¾ºù¿0tT¿g y¿Õ/~¿½Õg¿gó=¿Mõ¿ý¸¾>z½ê!>-·>~­?9,?>KI?æU_?o?®(y?H~?Êü?46~?áy?¹s?SÃk?ãb?ÃGY?N9O?gñD?Ó:?@b0?ôY&?N?	/?S%
?¹?Æò>xþâ>à>Ô>´SÆ>°7¹>#ä¬>_Q¡>w>xL>·È>ÄÅs>U$c>êS>ÑE>ö7>òô*>0>m7>Âý	>v >\,ï=Û¢Þ=ã;Ï=ñãÀ=¼³=$§=%=É½=µ=æ»z=yXi= )Y=©J=P<=§/=á"=ë=ó=»D=®Pô<|[ã<xÓ<ÃãÄ<ð8·<éª<Ùª<§<g<3º<Yùm<øs]<-N<Å?<Du2<&<<_Ï<jÓ<Íù;Çç;¯×;¶È;¬]f?'/o¿?{Þ½¾§>,Zø¾çZF?Ì`~¿²¶M?ë¼×^¿?D?Ìê>n¿ëÑÄ¾Û!^?sÊ-?<Î÷¾J¿~¿ÄÁ¾¯Ð?Eû?£Æ&?uÜÒ½çA¿þ¿ ;L¿O;¾>Ã'4?á¢t?Ð|?½¯U?Ó?%c>3Qó½&Ù¾¬+¿ÌéX¿Xt¿Ýz¿Ç|¿.o¿¾Y¿ùg=¿¸*¿þAú¾ÌR·¾B%k¾ÑÙ½ë&"<¥ñ=º^>°¡>Ç>Ð"í>ï^?^P?y#?U>/?9?±B?$J?nuQ?ÓW?ÎÒ\?+qa?-we?»÷h?l?©n?öp?ör?$²t?3v?áw?â£x?^y?iyz?l6{??Ú{?:h|?Bã|?ÝM}?9ª}?>ú}??~?{~?¯~?­Ü~?³?%?ÅB?\?r???Ô£?.°?àº?$Ä?*Ì?Ó? Ù?VÞ?Ùâ?Áæ?$ê?í?ï?Îñ?µó?[õ?Èö?ø?ù?ú?Ñú?û?ü?¢ü?ý?zý?Ðý?þ?\þ?þ?Åþ?ç,~?N	¿'ö!½NÀ>@ð¾g}·>É<½·ñ¾	n?Âä_¿ã÷<Æ¶i?³×¿2¿E9?6B?5ù¾VSz¿(2¾LâL?ánm? l>ñ!¿®¿ëÎF¿ªn0¾hò>¬e?,,~?ÇJ?ô8Î>N½Vàí¾GCF¿Ès¿mÏ¿WQo¿tI¿i¿¿õµ¾í$õ½-Ù=ù¯>hû>0$?ïB?GZ?tÆk?ø,w?ü¢}?ûõ?æ~?Ü {?Â>u?ÿÂm?me?2§[?Ë®Q?)pG?È=?×2?ÐÀ(?Ní?m?LK?W?üzö>T¯æ>Ð»×>·É>1P¼>ÝÌ¯>?¤>#>ã±>¡>áöw>õg>,DW>¯H>DÃ:>Xí->õ!>JÌ>¢e>n´>Yó=´â=YÛÒ=CÄ=ð¬¶=çª=>>=½F=-= =Qom=÷\=£M=d_?=?2=©¼%=_<=7==ø<UXç<}I×<ÝWÈ<°oº<~­<^s¡<&><Ð<Q<&r<Va<¬±Q<Ù"C<»5<_û(<@<=U<q,<hpý;ù×ë;BxÛ;;Ì;·ô=6X¿¿Ì?Am¿Ób?Üþn¿ê×?ï¬a¿Ì¹>UAø>â¿(ðÐ>D÷G?ªï7¿$Q0¿'?
¡_?ÿgV¾|¿ì|¿{n¿>Óy?8àJ?Aå³=µG!¿é+|¿r}a¿W&â¾@Eô=ýA?ñPj?¬?h±b??ó!?J>³¬½gÍµ¾ö¿&¢O¿ÍIo¿Ï(~¿û~¿ö:s¿!_¿D¿6ñ%¿3¿³uÇ¾E¾ Ò
¾(?¼â¿½=~ÑF>+Ø>Ë½>ýkä>]|?EÝ?& ?%,?ª+7?h@?À±H?pÝO?k"V?Õ[?b`?"d?T*h?öPk?Dn?«op?÷r?Lt?TÛu?^5w?ax?Õey?Gz?.{?Å´{?ÀG|?Ç|?{5}?}?ñç}?µ/~?àm~?º£~?_Ò~?Æú~?Ä?<?SV?m?¿?Í? ?[­?n¸?Â?TÊ?Ñ?À×?%Ý?Ñá?Ýå?]é?fì?ï?Mñ?Eó?úô?tö?¼÷?Øø?Íù?¢ú?Zû?úû?ü?ûü?cý?½ý?
þ?Mþ?þ?ºþ?C#?L¿J<?%Ñ¾Y>"¦¾«?Ô­c¿S{?Eæ¿+Ïá¾½Ü?±÷¾¦Üj¿{ÌÜ>n?FC¾¿ÏÞ¾?*F}?@à>j÷Þ¾ðv¿Ê3`¿ª¾^]ª>Ú+S?Äù?\?=l?eYi=Ãl¿¾([6¿ÿl¿üÅ¿tu¿úS¿ëç"¿ Ò¾3¾{]=Ñ>ÿñæ>0¨?F<?	U?/@h?üñt?w|?	¾?¢k?<|?7Îv?©o?h@g?Ùô]?T?ÏâI?Ì??ÕD5?!+?á;!?¯¦?­m?
?v[ú>Y\ê>£5Û>6åÌ>«f¿>ò³²>ËÅ¦>%>p>ØC>Ý&|>¬øj>±ëZ>öîK>ò=>\å0>»¹$>æ`>MÍ>ò>a÷=Wjæ=¤zÖ=(£Ç=Ñ¹=ò¬=C÷ =¢Ï=;m=Â=q=ÖÅ`=n.Q=q«B=Ï)5=È(=Îä=y=kß=âü<)Uë<ÿÚ<õËË<o¦½<8|°<â;¤<7Õ<9<Y<ÇRv<9e<+OU<F<1¸8<2å+<ö<Û<x
<ç <ëèï;ð@ß;ûÀÏ;eE¿1©½Ýq-?Òªi¿¦Lw?,r¿.Q?Fê¾ÐB¾ªg[?ñÁe¿QY½ßw?b¹Ë¾Éøf¿q¹>FJ{?Ävª=Z¨f¿µ-H¿L	>#f?\sf?>d#÷¾ùuq¿iq¿D·¿Ñb¼õ?üß\??:om?2¬3?SßÅ>ûN,=¶n¾¼¿ozE¿g[i¿	|¿ ¿Ôv¿Lzd¿²=K¿Kv-¿¿¥]×¾EÕ¾{(¾#»7½º=£ñ.>Xü>+ý³>ÎÛ>Jÿ>à_?i?ãÒ)?ûÄ4?e>?ØF?N?N?¸T?þ_Z?ÂN_?Jc?Yg?hj?tpm?æo?ç	r?%ås?¤u?çv?
x?O+y?Êz?3ßz?§{?·&|?}ª|?­}?}?SÕ}?~?ç_~? ~?äÇ~?²ñ~?ç?D5?mP?óg?S|?ø?@?{ª?ñµ?Þ¿?vÈ?èÏ?ZÖ?ïÛ?Äà?ôä?è?¸ë?pî?Êð?Ôò?ô?ö?r÷?ø?ù?rú?1û?Öû?eü?àü?Lý?¨ý?ùý?>þ?{þ?¯þ?#¾#¶'¿:~?ýp¿*ùX?Z[¿ t?Û}¿Ï§:?
H¼uKO¿\b?hH,>¿ö¿ ¥=Nà?|ý=÷m¿(K)¿hìÏ>cð~?ÑÇ?Àx¾9d¿mr¿,Q÷¾Ü<>÷V<?Ç;}?Rój?«?¦'>¾ß$¿öªb¿~¿¢6z¿x]¿`/¿kTî¾tåk¾®©ë:9A_>Ò>Wó?ÄR5?àPP?Bud?Mxr?ù'{?ÿT?´Æ?Ô3}?é>x?#uq?ÅNi?0`?mV?4IL?½ùA?Í©7?}z-?ó#?ÑÛ?i?Ç?"7þ>zî>P¬Þ>'*Ð>{Â>[µ>ý}©>!>z>X>Ù*>ván>w^>¢WO>o A>üÜ3>~'>Bõ>Ä4>/>þ²û=ÂMê=ÁÚ=Ë=õ¼=(ß¯=7°£=xX=>É=Óô=Ñu=d=A¹T=u÷E=Y;8=ár+=9=¶z=¿,
=¹ =úQï<~µÞ<
@Ï<,ÝÀ<Ýy³<e§<Fl<'¢<¾<}z<i<©ìX<XÝI<§Ù;<Ï.<'¬"<ø`<Þ<Ï<Þùó;	ã;iFÓ;VÑs¿éiA?Vbø½Ú¿¬¾ÅÜ? ü¿G»>4>³3/¿hõ?T7¿¦"ï¾ÌY|?Ð«¼¶*¿¯¿þ<ê~?pÅ¼>Ê@¿©ði¿³Cº½õH?jWx?}ñç>òë£¾+'`¿{¿ãc-¿¤"¾·PË>Ö}L?³|?¸Íu?×C?çøí>/ú=}X¾}f ¿}:¿ëb¿Sy¿åÿ¿t×y¿`yi¿TQ¿·4¿øË¿íç¾o?¤¾AF¾¶Ù½Ý<+=ø>#z>Õª>¿Ò>4/÷>oØ?uH?³'?V2?&B<?{øD?L?2IS?OY?7^?©§b?f?îâi?Ðl?8[o?âq?|s?u&u?sv?lÙw?Îïx?3áy?z²z?ãg{?|?a|?s}?Ái}?cÂ}?-~?²Q~?Q~?:½~?vè~?è?V.?mJ?Áb?Ów??à?§?i³?­½?Æ?CÎ?íÔ?³Ú?³ß?ä?Çç?ë?Öí?Eð?aò?4ô?Éõ?'÷?Wø?^ù?Aú?û?±û?Eü?Åü?4ý?ý?çý?/þ?mþ?£þ?Sw¿â±[>Xý?2e¿_{?â|¿^$q?ä>¿ >@Ú÷>Ç}}¿Ä*?C?'n¿±¾³ås?Ë<Ú>\H¿°êV¿Ø&>°Ur?Ç¯G?A#½J¿â2}¿·ä¿XÃÿ<«¥!?¯þu?û¹u?ê0?Ï>Ï:¾Àõ¿W¿»z¿¦}¿âe¿;¿}¡¿±Ã¾
äN½ìW.>µ£¼>û	?.?¾J?Îf`?Ào?Îy?òº~?B÷?µ~?¨y?5's?SGk?Zb?d¶X?1£N?}YD?i:?Í/?pÈ%?g?q§?£	?÷?¥ªñ>Çâ>lÓ>bÅ>}¸>Ð4¬>ß¬ >ãÜ> ¼>¯A>OÉr>{8b>±¿R>HND>5Ô6>B*>[>>ðl	>Ußÿ=ö0î=²¸Ý=ÐaÎ=ÞÀ=¥Ë²=i¦=?á=4%=	'=z³y=Xbh=
DX=pCI=ÜL;=ôM.= 5"=ðó=z=+º=ÇNó<zkâ<´Ò<çÄ<w¶<æÌ©<U<1<ôÕ<2¬~<þl<&\<:M<û><×¸1<3b%<Õæ<7<F<Ï
ø;KÒæ;ÖËÖ;æ%¾í	z?Z­U¿Aä>éØA¾}C>¹ã«¾}o+?Ä½w¿M`?@¾2N¿UT?ÈÒ»>¶u¿¨¾_g?>Y?¾¿|¿E¥¾ø0 ?ìË?V5?ó¾¸H¿à¿F¿[½¾ñ>^b9?øv?{·{?5PR?ö$
?C²N>æ¾yJá¾·.¿°öZ¿úgu¿X¬¿'C|¿¼ým¿¡W¿{°;¿M¿ëiö¾³¾¼ºc¾z¹Ë½Ð<§Óý=g&d>"( >¯ÊÉ>,ï>/G?w?»C$?ß/?X:?ºC?ËðJ?tÔQ?ÑÙW?4]?D°a?®e?'i?&-l?¢Ín?çq?Os?ÊÉt?Hv?®w?R³x?Á¬y?z?{@{?÷â{?Èo|?Îé|?S}?#¯}?þ}?@C~?Í~~?c²~?ß~?Æ?K'?TD?y]?@s??q?¤?Ö°?r»?¢Ä?Ì?{Ó?rÙ?Þ?ã?÷æ?Rê?:í?¾ï?ìñ?Ïó?qõ?Ûö?ø?%ù?ú?Üú?û?%ü?©ü?ý?ý?Õý?þ?`þ?þ?ò:=¿äTi?Ô¾Hç¾ôé?/(¿Ö?÷,l¾R
¾II[?Áp¿Ô5>íY?´"8¿Û¿L?¿¹/?b¿t¿5»½¬*X?uèe?¯(>(¿üâ¿{É=¿rËú½
¤?acj?-¹|?#»D?4½>Ç*ª½.û¾¨ÅJ¿ÛÁu¿Í}¿û-m¿F¿Ñ¿°­¾0JÒ½	ú=8â¦>Å ?µ&?$çD?\?nËl?OÎw?ÿï}?Bý?µ~?KÃz?¿t?â)m?Hqd?ðZ?¤ðP?é®F?Z<?Î2?E(?_8?¸¾?1¤?âï?ÌKõ>ýå>6¬Ö>È>_»>>ê®>7£>Á>>,÷>íW>2°v>¹Ýe> 'V> {G>Ë9>Ñ->0!>>ª>²>ðò=uWá=íÀÑ=<Ã=
¸µ=ã!©=öi==6Y=Ê}=0l=ÈÎ[=cL=X^>=)1=Þ$='m=^Ç=Þ=K÷<s!æ<-(Ö< JÇ<"u¹<f¬<b <9t<)<sl<àp<¢'`<ÔP<B<©¢4<>(<²l<<iu<Áü;øê;CQÚ;÷j,?û£Ò>]Öu¿TQt?ÉN¿U A?ÀDV¿y?ûs¿À?®>î{¿e ?8Ù1?"ÚK¿@¡¿¢):?£R?F¾ýí~¿b$	¿%2á>±|?½¸@?ô½ó<Î+¿Á~¿m[¿åÍ¾ªO#>¼Î#?òËm?b?e÷^?pD? a>É½;¯À¾ä4"¿ÒR¿ìp¿¥~¿½~¿r¿A]¿ù_B¿Ó#¿Â¿ºÂ¾´¾¹¾ß¼Í=;N>%#>hÃÀ>¦ç>b¬?Në?!o!?þ_-?5å7?P$A?@I?WZP?V?û[?µ`?kÓd?<ih?«k?Ñ=n?úp?Õ¤r?¢kt?Yöu?ÎLw?Ûux?uwy?ÏVz?n{?AÀ{?³Q|?½Ï|?ó<}?}?í}?4~?r~?_§~?Õ~?ý~?# ? >?X?n??ô?¡?9®?.¹?¬Â?äÊ?Ò?,Ø?Ý?"â?#æ?é?ì?4ï?tñ?hó?õ?ö?Ò÷?ëø?Þù?°ú?fû?ü?ü?ý?jý?Ãý?þ?Rþ?þ?K_+>qÙ_?oi¿Kö>ú
ô½:e¼ï½Sä¹>fªC¿Rô?h,¿+Û¾ÿ}?@Ì¾bqY¿¼Ð?âç`?j_¾è¿3É­¾Jæ1?¡úx?î¹>¡Õ ¿z¿Ï~W¿æ¾ßÅ> Z?]Õ?KV?ï>N½<Ñ¾;<¿^0o¿Sû¿éQs¿`)P¿¿íÇ¾|G¾Ë=É>Ï§î>³×?vÌ>?,W?±i?âÎu?Mô|?µØ?«>?¬Ö{?m<v?Fön?vf?o]?g1S?äùH?¦>?{]4?]>*?«_ ?1Ò?Æ¡?JÖ?ßèø>åüè>>éÙ>«Ë>M?¾>A±>Á¥>²>x1>m>z>.i>îY>u¨J>oÁ<>0É/>Á°#>ãi>ç>>±öõ=	öä=ãÕ=>`Æ=W¤¸=Ú«=ò=úÜ=X=Mð=¦þo={Y_=MÛO=ÍoA=4=_'=Zæ=©==VHû<i×é<;Ù<WÊ<Ãr¼<å]¯<n1£<AÝ<]R<Ì<Ãt<Åc<õS<>E<z7<IÎ*<ò<é<¶¤	<Y <¥cî;°ÖÝ;¸c|?$jø¾xÃÑ¾%X`?-~¿5ô?ó]¿ñn?$%¿¥<×=?Ùgv¿¢>e¿j?
¿ÃÊU¿nô>£Äs?£ÔÚ¼QÍp¿8¿±,n>%n?÷7]?pS>Ó1
¿Bv¿l¿r2¿Û¹=?Ab?¼ö?¦°i?8-?¥z¶>Õ±C<á#¾6¿.`I¿­k¿bì|¿N¿u¿ø|b¿ÚÂH¿<®*¿v)
¿rÑ¾¾~|¾Ñ½å1=^â7>ó>ª·>{ìÞ>H?)¯?
?Ù*?Ñ«5?N0??HG?äÚN?@U?¬ÖZ?<¶_?lõc?
¨g?¥ßj?Å«m?p?¨6r?ÿt?[£u?Ïw?i7x?OAy?Ý'z?¼ïz?û{?"3|?@µ|? &}?°}?RÜ}?¤%~?(e~?-~?ÔË~?õ~?Ü?Ó7?¥R?ßi?ý}?i??«?á¶?­À?*É?Ð?àÖ?cÜ?)á?Kå?àè?ùë?¨î?ûð?ÿò?½ô??ö?÷?°ø?«ù?ú?@û?ãû?pü?êü?Tý?°ý?ÿý?Dþ?þ?Ôk?Is>2g¿=w?$C¿y¾ ?¢¿(¿ÆT?µ}¿Í`?·X}¾~L*¿¬«u?(Ù¼³Z{¿v>@È{?ßæ¼û8x¿ÔÉ¿Û¬?ù?'z?Íñ¨¾x8m¿âPk¿¥ÉÕ¾µg>øF?K?e?ÃW?Y±î=Øµ¤¾+Ï,¿wg¿¿¡Fx¿dY¿ï)¿¥â¾ñúR¾_ÈÌ<mÇt>ßXÛ>Ø?¦p8? ²R?(,f?øs?È{? ?£?¨Ê|?>w?S¬p?hh?I5_?YeU?O:K? é@?6?¥p,?8"?Íá?8?$º?Ïü>sfì>#Ý>a·Î>ÂÁ>ÕP´>xI¨>³ÿ>k>>{~>Ö%m>ô\>ÆÔM>m·?>92>D&>~Ð>æ#>V1>6Ùù=mè=´~Ø=ÀÉ=»=A®=4{¢=Ê8=p½=û=¹Ìs=#äb=/'S=;D=ß6=¸.*=_=ña=r'	=Eÿ<\í<GÝ<¸Í<ap¿<b&²<zÈ¥<HF<<%<¢¥x<bg<NRW<y_H<Kv:<T-<kx!<B<Ô<Ò<R,ò;\á;¡È>·}¿SÜÛ>GÍ>m4%¿ð=G?q@¿4z?`W¾s÷¾
x?¸#?¿¼ø¾xõ?Y2B¾¦x¿M9>÷?<}z> S¿º\¿Kf<ßªV?m©q?{À>É¾lh¿Ñw¿â!¿H´½rÙä>T?^A~?8dr?uß<?eÜ>Ñ±=p¤y¾º0¿Wv?¿a¯e¿åz¿ë¿x¿=Qg¿ÖN¿a1¿g¿Ùà¾³n¾°!9¾·v½0wY=!>¢é>¥®>E±Ö>F¶ú><j?¬?ZJ(?Dk3?Æ5=?'ÎE?&VM?¼ìS?®Y?¢³^?&c?ôãf?5j?m?Go?ÈÆq?àªs?Ou?°»v?ý÷w?O
y?-øy?fÆz?'y{?|?X|?°}?}s}?ÓÊ}?{~?X~?Í~?úÁ~?ì~?x?l1?M?e?Ôy?Î?`?Ü¨?´?§¾?iÇ?þÎ?Õ?@Û?-à?qä?"è?Uë?î?ð?ò?aô?ïõ?I÷?tø?wù?Wú?û?Áû?Sü?Ñü?>ý?ý?ïý?6þ?sþ?«S?æ7¿	¾Ã[?#³¿F1{?|Ïz¿"ý?PXi¿4K?bp>éÊl¿5¯B?"J»>-|¿}ÃÆ½°}?ÿ>Ù	^¿UA¿¶b>Az?éQ4?#¾tX¿µx¿úN¿·9á=$É/?6Cz?Ð©p?âC%?jH\>æm¾Ó¿£r]¿º¦|¿÷|¿µa¿K05¿»û¾C¾Â¼(|G>¦Ç> ?âÕ1?Ë¢M?´b?-q?wkz??¬ã? }?Uçx?àKr?QGj?|@a?VW?
pM?#C?hÑ8?
.?ø$?í?{?i?F ?Ìï> [à>õÀÑ>júÃ>ò·>«Ðª>À^>É£>ÿ>|/>°Èp>Y`> Q>ý¬B>êN5>×(>á6>`>ñF>»ý=¡2ì=^ÝÛ=#§Ì=¦|¾=ÒK±=º¥==}ï=º=½w=Ànf=sV=¢G=
º9=×,=´Ø =6¯=ÚK=ë =LCñ<Oà<¿îÐ<þmÂ<ßî´<_¨<N¯<ÃÎ<}¯<%|< k<¯Z<íK<`=<^:0<Gþ#<<O<J'<ÿôõ;áä;úü¿Ùë2¿®Tw?å`¿RF=©E>Q#M¾ZN¼ÐÒ>Ý*[¿¡Øx?>Â¾<&¿îAn?£§1>Ê~¿ÕC	¾Á(v?'Õþ>+Î'¿\u¿¥R¾M·5?N}?¥¯?¬¬r¾gwU¿~r~¿:¿WW¾|®>C?¦z?Óþx?rK?®p ?ÿ%>Ê3¾±ñ¾Ù4¿Í÷^¿Õgw¿í¿g{¿7¼k¿ÐT¿í%8¿¦z¿<î¾ó²«¾X£T¾x¯½Öð<§B>no>ÞD¥>®dÎ>gJó>¹
? ¿?û³%?¥#1?È4;?-D?&ÌK?FR?VX?U­]?/b?ýf? i?l?o?6Uq?GHs?ùt?qqv?·w?vÒx?ÁÇy?kz?ÅT{?ô{?|?÷|?ù^}?¹}?~?°J~??~?ø·~?èã~?ö	?ì*?xG?1`?u?&?5?¦?*²?¼?¡Å?sÍ?:Ô?Ú?,ß?ã?bç?¯ê?í?ð?(ò?ô?õ?÷?7ø?Bù?)ú?òú?û?5ü?¸ü?(ý?ý?Þý?'þ?gþ?ÎÙ¼Ý|¿/?ãô`>½©2¿Ï3`?{Åf¿íJQ?:F¿ù¢<½)?¨|¿ -Û>`¤1?
¢[¿Ü³Ø¾Kpf?	¬
?ÀN3¿ªce¿rìh= üh?ÞU?ª<=ùÜ<¿´M¿ú`,¿Å½{?Ïªq?Ùy?éG9?Y>QR¾­	¿ßcR¿eÙx¿>~¿i¿«Ç?¿
¿Û@¾¥!½
È>o³>¶)?jþ*?-VH?E¡^?®n?ÏÞx?l~?ðþ?ûS~?z?ÇÔs?l?å;c?=¦Y?ùO?TTE?| ;?yÃ0?Ú¸&?:õ??z
?Ó?L/ó>äã>FÈÔ><ÕÆ>±¹>ªV­>Ö¼¡>ÆÛ>Æª>ð >·jt>q¾c>Ð+T>¢E>B8>Íi+>	>û>f\
>ÇÎ >¥Ðï=à;ß=fÊÏ=§hÁ=M´=0§=Að=!=ä=´h{=Rùi=×¾Y=¤J=<=\/=ÜQ#=xü=?p=H=8ùô<Vøã<p%Ô<kÅ<Y··<öª<S<ö<ÕÅ<S5<n<Æ^<`¢N<ìI@<ið2<#&<¥ô<2<Â/<«½ù;öfè;åè¿Í>UR?¿y¿iX7?Q#÷¾¥¢Ý>sk¿1}U?/ó¿ Ì??®=¢[g¿°U8?ú?ðg¿ß¾?0W?·6?JMã¾ó¿+Ô¾é ?ëº?âÑ,?¡÷½D=¿Ü¿WÓO¿âé¨¾Ük>®§0?qBs?×q}?ÃÓW?gÙ?íUp>'Ú½6ËÓ¾È)¿ÅW¿_ s¿YT¿Ã}¿ä»o¿:Z¿>¿®`¿Âü¾#Ñ¹¾/üo¾ùã½?pº;¹®é=ðZ>£ú>bÆ>{Íë>ÔÆ?]É?#?
Õ.?e-9?hDB?î<J?%7Q?lPW?W£\?ÎGa?(Se?dØh?Tèk?Ón?ôáp?3är?£¢t?&v?5vw?Ãx?y?Íqz?Ó/{?Ô{?Fc|?÷Þ|?%J}?§}?s÷}?%=~?y~?Ï­~?Û~?V?Q$?ÀA?=[?Pq?o?þ?U£?À¯?º?ÑÃ?âË?ÞÒ?ëØ?(Þ?±â?æ?ê?÷ì?ï?ºñ?¤ó?Lõ?»ö?ù÷?ù?ûù?Êú?|û?ü?ü?ý?vý?Íý?þ?Zþ?[[¿êî¾¹~?9¿á¸¼ÙÈ¸>îé¾î°>93¶¼ð÷¾Fo?ª^¿,M©<¢j?ý.¿w3¿ÔU8? C?Çmö¾t¡z¿7¾KL?+àm?Ìop>zP¿¦ì~¿^G¿_´3¾2ñ>íZe?d@~?»(K?}HÏ>¨ùE½üì¾÷E¿ý¥s¿TÓ¿~so¿Ñ«I¿¿;¶¾áb÷½.×=9>û>ì#?XÎB?ØZ?mµk?a"w?Ù}?[õ?"é~?Å&{?âFu?ÆÌm?a'e?î²[?ýºQ?|G?1(=?ßã2?ÎÌ(?ðø?Hx?V??~ö>ÑÁæ>JÍ×>3®É>·_¼>sÛ¯>ð¤>ù>è½>à>éx>"g>VW>ÑH>@Ó:>Aü->ö">?Ù>µq>¯¿>vnó=:â=íÒ=TÄ=´¼¶=ª=éK=vS==6=Øm=
]=YµM=ño?=¦'2= Ë%=·I=¢=£=!¯ø<Ylç<\×<3iÈ<Óº<­<W¡<'K<,Ü<&<;r<ja<ÒÃQ<¼3C<s¦5<ÿ	)<«M<ça<;8<Xý;cìë;¿§b?)Ì=U¿Oï?ü¾n¿ÅÎc?:p¿Êï?@:`¿ÜP´>MÖü>ò¿¶ÆÌ>ãGI?6¿;¤1¿·%?¾e`?åP¾[Ú{¿¿¦8½>Õ×x?ÁK?»=ú ¿î|¿áÚa¿pã¾Ñóî=Ã?j?g³?íb?ÌO"?>ã4½,µ¾¯¿JqO¿(.o¿ ~¿~¿sNs¿³_¿F¢D¿&¿[¿"ÆÇ¾z¾ßg¾#¥¼»¼=èYF>Ô¡>½>@ä>Àh?ÙË?ºp ?,?±7?äv@?¨H?`ÕO?_V?¯[?Â\`?wd?E&h?nMk?2n? mp?¦~r?}Jt?Ùu?Û3w?7`x?±dy?Fz?S
{?´{?G|?Æ|?ÿ4}?®}?ç}?e/~?m~?~£~?+Ò~?ú~??ñ;?6V?öl?©?º? ?M­?b¸?ûÁ?KÊ?}Ñ?¹×?Ý?Ìá?Øå?Zé?cì?ï?Jñ?Có?øô?sö?º÷?Öø?Ìù?¡ú?Yû?ùû?ü?ûü?bý?¼ý?
þ?Mþ?Þe¿áVÝ>z8?Ö{¿5*?:¥¾*ò2>³Ç~¾ñ£?e[¿ ï}?Ë¿öÝÇ¾ô?-¿¾f¿%çï>vÍj?Ág¾6¯¿üÏ¾4X%?+|?ÆÔ>é»é¾w¿D¤]¿zê ¾wØ²>	U?îÿ?±Z?müü>7=ØÄ¾GB8¿ÿm¿ Ü¿Òt¿}ÓR¿o!¿®KÏ¾û-¾v=(>ÈOé>È¢?ì=?|2V?íªh?6u?o¥|?íÆ?^?â|?¢v?³ro?Ìg?I²]?ûÏS?¹I?qH??*þ4?ÅÛ*?ø ?»e?W/?~]?$êù>Þðé>úÏÚ>GÌ>R¿> _²>v¦>]I>dÐ>I>B¬{>j>®Z>K>á=>j0>¨h$>R>Ý>w°>÷=løå=Ö=[@Ç=u¹=é¬=§ =c=(=;=Rq=YV`=ªÆP=ÜJB=ìÏ4=D(=ò=¹=û=eü<Zàê<ËÚ<ËfË<KH½<$°<Zê£<X<ò<Ó<}Øu<<Çd<DåT<F<|\8<Ú+<°¦<3<³@
<§ <Ïqï;mZá>©×f?~1¿ôî7¾8??·Mr¿|?ów¿|[?¡}¿ÅÄ¾QS?`°k¿t<wt?à¾(b¿`ÿË>y[y?QK=j¿åpC¿×w0>åh?Äc?û:>÷ ¿s¿Âßo¿â;¿Nè¿:b?Õ^?¾?âRl?¸1?þ.Á>õ=²£¾à7¿ë®F¿Oj¿¦Q|¿¿Crv¿:àc¿Ï|J¿l,¿Ã2¿ØÕ¾9¾5%¾è*½Å¯=U®1>P;>_µ>¢Ü>² ?Æ?Ä?=#*?¼5?°£>?G? oN?7âT?aZ?|n_?í¶c?¤qg?U°j?¥m?^öo? r?ñs?úu?ðv?Ò%x?2y?£z?Eäz?{?*|?Ê­|?}?}?x×}?p!~?a~?~?É~?¾ò~?Ï?6?Q?h?Õ|?i?¡?Ðª?;¶?À?­È?Ð?Ö?Ü?ãà?å?«è?Ìë?î?Ùð?áò?£ô?)ö?{÷?ø?ù?xú?5û?Úû?hü?ãü?Ný?«ý?ûý?@þ?Më½÷<{?SÖz½O¿Cá~?ÒþZ¿æ<?¦KA¿nd?þñ¿z<Q?¯ÿ¾¦;¿À\n?¨=PT~¿B°.>K~?.w:=b9t¿ªØ¿oí>Þ?M?«w¾Íxi¿«n¿F®ä¾ñb>dB?V~?v´g?¡?M>Ù:¾ÞZ)¿Ý,e¿§~¿`)y¿"6[¿X,¿Øç¾ó]¾2yr<kTk>@×>#?7?K¨Q?Ùle?s?{?­s?´?Æù|?0æw?=q?Îh?.¤_?×ÙU?³±K?'aA?G7?¯å,?ô"?ÓO?ç?+?0Bý>í>NÐÝ>rZÏ>a·Á>Má´>#Ñ¨>ñ~>8â>+ò>¿K>äèm>Hª]>Ý~N>&V@>H 3>Î&>5Q>Ý> ¡>©ú=tVé=k3Ù=,Ê=B-¼=,%¯=£=C·=,3=!i=Át=¢c=ò×S=Á%E=,x7=;½*=+ä=aÝ=R	=u =XTî<uÉÝ<adÎ<ÁÀ<¡»²<]S¦<Ç<Ø<	<õuy<u$h<¶X<[I<;<¶.<µÿ!<À<*I<Ø<;÷ò;ÕN~?9D>ÿ¿ªÕ?m^¿=¿4´-?Ù'¿ ç>ä^©¼õ¿3}?Q#.¿mÁº¾®x?é½U?|¿Îì=¼?>K¿²äb¿î¯½xP?BÙt?Ò>/¹¾äe¿B¢y¿n&¿Èþè½a´Ù>\ÓP?}?^ós?ïù??ý)ä>×Ñ=wk¾8¿ÀK=¿fVd¿	êy¿Óø¿æ%y¿îFh¿;P¿9ð2¿æ¿8(ã¾¼n ¾Ôç>¾õò½=%E=ï>ð>¬>¦õÔ>¾)ù>Ô¹?C?<À'?ñ2?ÙÊ<?apE?M?ø¤S?poY?ÿ|^?äb?ºf?
j?+øl?~o?"¯q?Xs?A=u?:¬v?êw?®þx?îy?¨½z?q{?|?¨|?Â	}?8o}?Ç}?G~??U~?e~?å¿~?Åê~?è?0?íK?d?óx??¸?J¨?´?9¾?
Ç?¬Î?HÕ?Û?÷ß?Bä?úç?2ë?üí?fð?~ò?Mô?ßõ?:÷?gø?lù?Nú?û?ºû?Mü?Ìü?:ý?ý?ëý?3þ?³F?K=?·ÊL¿O¾ÐJ?áý|¿ÛÏ~?xn~¿ÅÞ~?¨V`¿tÈç>åÓ>h|r¿8?G.Õ>y¿©À¾ðë{?éÁ >týX¿G2G¿>:Ðx?sx9?qòñ½üT¿Þz¿úà¿U¸=G,?ÇIy?r?F0(?õCj>¦æ`¾Z¿øû[¿44|¿Ts|¿èËb¿n¶6¿Àÿ¾¨,¾òù¼hA>lÜÄ>\q?4ä0?oèL?éûa?öÑp?Î6z?§û~?¶é?\º}?#y?Dr?ìj?a?uØW?h¾M?=rC?% 9?|ê.?së$?6?´Ù?Þ?LK ?/Fð><Îà>¬-Ò>ß`Ä>Vb·>5+«>°³>`ó>á>.u>ûJq>PÓ`>4rQ>C>Û±5>U3)>æ>¶°>¨>ºFþ=S´ì=*VÜ=£Í=hå¾=\­±=^¥=é=2>=ÿO=$#x=¶íf=3éV= H=h :=R6-=`1!=¼=§=dè=TÈñ< á<õaÑ<6ÙÂ<¦Rµ<^¼¨<·<-<Rú<k}<®k<'([<*ñK<È=<0<ºX$<Ëï<¢Q<-p<§|ö;5!"?Ö,¿a¿g}?76¿>MÅ=\â½|2À½µ¤ö>ûCd¿/,t?0'¤¾zo1¿Áh?.ùd>3R}¿6¾Õs?ìÕ?íÌ ¿kWw¿¦íp¾¸t0?5~??[¾^R¿ö~¿#[=¿Vñh¾¿¦>½ÿ@?ñ5y?Ó¿y?büL?Wî?°/>Õí)¾6{í¾`O3¿sø]¿¼êv¿}á¿!h{¿Rl¿ `U¿!9¿»t¿<ð¾t¯­¾WzX¾\^¶½¸¤Õ<ã>[l>¹ø£>Ì9Í>÷>ò>¤¥	?aU?V%?ZÑ0?mì:?³ÌC?K?©cR?âVX?M]?Vb?å f?oi?Åkl?o?,Eq?Z:s?iít?ôfv?}®w?Êx?êÀy?}z?¡O{?ð{?({|?ªó|?\}?¶}?é~?ÍH~?~?¶~?®â~?æ?*?¬F?_?u?¢?Ã?»¥?Ô±?M¼?`Å?;Í?	Ô?îÙ?ß?sã?Gç?ê?ví?òï?ò?öó?õ?ùö?.ø?;ù?#ú?ìú?û?1ü?´ü?%ý?ý?Üý?%þ?Qxs?°Æ½Â±y¿¶1?ÅÆ'>Uµ(¿ó×Y?îra¿6ÐJ?°¿±¼)0?}Ö~¿GMÍ>h®6?:+X¿CÅã¾2äc??¹½/¿®eg¿vj$=µMg?W?6s=­:¿±¿<m.¿Áý0½m?úêp?Òy?·:?Óð¢>¨	¾{Z¿Q¿x¿Ù¬~¿ºi¿x@¿rø
¿¾'½>¹*²>é?*?ôG?âX^? Zn?ýÀx?í^~?yÿ?_~?Í(z?©ïs?b3l? _c?ºËY?¿ÁO?{E?°';?ê0?Þ&?Ç?´ª?Â
?§ó?]ló>¾Éã>ïþÔ>ÆÇ>â¹>=­>ç¡>Ú>MÐ>D>\¬t>Åûc>eT>×E> C8>N+>eÈ>fÅ>
>ßñ >ð=Çxß=Ð=xÁ={5´=ú¹§=â=.I=Õ6=z­{=V9j=lúY=wÛJ=È<=f¯/=~#=&=ù=RÃ=L<õ<Â6ä<_Ô<ª¡Å<©é·<_%«<åC<5<ë<pX<çÞn<I^<øÚN<~@<k!3<¿±&<<Z<T<ú;:8¾ÚÊ~¿àa>¿EI?]|¿x@?lv¿8ò>½6¿áíZ?Vö¿ÂÚ9?JìÂ=©j¿W3?#	?èEe¿E@é¾¶BT?á%:?Û¾.Ë¿ÂgÛ¾C
?^?-/?}.~½Sa;¿ÊÂ¿S9Q¿y$­¾nºc>s>/?·°r?"­}?@ªX?I?ou>¢Ð½o§Ñ¾âÁ(¿îþV¿Us¿©C¿í7}¿ p¿0fZ¿Uõ>¿¡Û¿îÀý¾ÔÎº¾Þèq¾¯æ½S;ÿæ=©|Y>#R>oÅ>|Eë>8??zæ"?«.?{9?$B? J?QQ?»:W?k\?P7a?ÌDe?åËh?vÝk?an?ÀÙp?Ýr?tt?µ v?qw?ºx?y?Änz?2-{?@Ò{?La|?AÝ|?©H}?¸¥}?Vö}?.<~?®x~?­~?zÚ~?Ë?Ù#?WA?ãZ?q?+?Ã?"£?¯?[º?°Ã?ÅË?ÅÒ?ÕØ?Þ?¡â?æ?ùé?íì?{ï?²ñ?ó?Fõ?¶ö?õ÷?	ù?øù?Çú?zû?ü?ü?ý?uý?Ìý?þ?#õ>+[¿³í¾:À~?Ãm¿l±¼c¸>:é¾X°>Æ»°¼<S÷¾¦o??^¿³3¥<D®j?¦¿Ù3¿ÁD8?Õ.C?oHö¾}¥z¿(Ú7¾îL?æm?©p>{E¿xë~¿fG¿Dà3¾åøð>£Ve?pA~?Ñ-K?·VÏ>ùE½­ðì¾óE¿3¤s¿Ó¿Guo¿¶®I¿¿Ö¶¾ö÷½#j×=Ì2>Àýú>eê#? ÌB?Z?´k?Ó!w?}?Rõ?Jé~?'{?OGu?IÍm?ô'e?³[?¡»Q?7}G?Ø(=?ä2?oÍ(?ù?Þx?V?"?ö>ÉÂæ>5Î×>¯É>`¼>7Ü¯>§¤>¥>¾>w>x>¤#g>|WW>¶H>Ô:>	ý->±">íÙ>Wr>FÀ>oó=@â=}îÒ=qUÄ=½¶=Zª= L=!T=¤=Ä7=ëm=]=H¶M=Ðp?=u(2=ÀË%=jJ=I=>=B°ø<fmç<]×<jÈ<«º<^­<¡<ÕK<ÎÜ<+'<<r<ka<ÆÄQ<4C<E§5<Ã
)<aN<b<Ù8<~ý;¼w¿ýI¿£Èb?IÊ=cU¿ð?Òn¿Ìåc?¿Jp¿¿ð?&`¿ò	´>ý>¬ò¿ÀÌ>oYI?ù|6¿ðµ1¿*¦%?þo`?DRP¾×{¿7¿ö½>WÔx?%K?ð»=¦ ¿ù|¿Ãßa¿Ûã¾q¬î=é¼?Ýj?Á³?ðb?¥T"?)>Þ½¸µ¾Ï«¿ºnO¿µ,o¿©~¿w~¿xOs¿3!_¿¤D¿&¿0]¿YÊÇ¾¾ºo¾(à¼V­¼=£SF>û>{½>¸=ä>¹g?ïÊ?ëo ?Ó~,?7?Wv@?¨H?ôÔO?V?][?z\`?8d?&h??Mk?	n?Ýlp?~r?bJt?~Ùu?Ç3w?%`x?¢dy?}Fz?H
{?þ³{?G|?Æ|?ù4}?©}?ç}?`/~?m~?z£~?(Ò~?ú~??ð;?4V?ôl?¨?¹? ?L­?a¸?úÁ?JÊ?}Ñ?¹×?Ý?Ëá?Øå?Yé?cì?ï?Jñ?Có?øô?sö?º÷?Öø?Ìù?¡ú?Yû?ùû?ü?ûü?bý?¼ý?
þ?í@-¿m¿0¿>E£B?íx¿4 ?ø¾ë>í¬U¾l=ö>¯-V¿¬ß~?Y@¿¨8º¾3»?Uâ£¾cc¿où>µh?Zz¾âÝ¿ òÇ¾ÑD(?é{?2wÎ>£>ï¾ÂLx¿H\¿¸'¾04·>@´V?$ý?7¶Y?Kñù>}=´¢Ç¾Á;9¿½m¿"æ¿}t¿¹9R¿¡« ¿\Í¾!¬)¾:]=ü>ê>Ö#?Ds=?ÏV?³áh?Yu?¹·|?DË?W?â|?v?Vo?Þâf?Ñ]?÷«S?òvI?#?? Ù4?ø·*?ÌÕ ?*D?!?¹>?¯ù>W¹é>vÚ>¹SÌ>¦Ý¾> 3²>ÙL¦>½">6¬>nà>ïl{>ìJj>jIZ>yWK>½d=>a0>É>$>Jî>|b>>íÌö=½å=¿ÙÕ=TÇ=E¹=¬p¬=R~ =
_=k=a=wÐp=Æ`=P=ûB=¡4=ë(=½n==(y=5$ü<¤ê<§ZÚ<2Ë<¬½<]÷¯<?À£<)b<Î<åõ<Vu<vd<®T<§êE<-8<Èc+<¬}<k<.
<u < w<¿âzÁ>xfm?J&¿Ûo¾E­G?%öu¿Ë}?U]z¿s`?:¿Îå¿½fN?wn¿£;=r?J6ë¾z_¿ÛbÕ>*;x?&=¬k¿Ôí@¿ß>>=Cj?-Ub?ÖBy>gm¿Ís¿æo¿·k¿<­?|l_?ÍÏ?î¼k?m´0?.Á¾>È#æ<ËÎ¾ ¿#MG¿®rj¿v|¿ }¿ö>v¿c¿¹J¿-,¿Y¼¿Ô¾$¾ s#¾I$½OÃ=3>ôß>²µ>(Ý>R> ? û?	ó?±L*?<05?ÀÃ>?+G?N?·÷T?%Z?Ù~_?-Åc?~g?»j?m?þo?´r?4÷s?Ou?'õv?Õ)x?5y?§z?âæz?P{?|,|?~¯|?!}?]}?Ø}?f"~?Xb~?½~?¹É~?Hó~?G?t6?uQ?Øh?}?£?Ô?üª?`¶?>À?ÊÈ?0Ð?Ö?%Ü?óà?å?·è?Öë?î?áð?èò?©ô?.ö?÷?£ø? ù?zú?8û?Üû?jü?åü?Pý?¬ý?üý?2|¿p¾uær?ö.m=â^¿È{?¶DM¿Ê+?ê­2¿RÏZ?æ~¿²¥Z?JÙU¾:1¿s?
`e<¯|¿koZ>²ã|?s <Äv¿V"¿Éú>ÿ?I?òh¡¾ªÍk¿Þ¦l¿{Û¾\<u>·5E?ÍÄ~?¼f?1?Ð;ÿ=		¡¾iy+¿<Wf¿ä~¿Q x¿Z¿
à*¿4ä¾>W¾½U¬<q>3ÂÙ>Ô.?jé7?KR?(âe?hs?~­{?R?)ª?Ý|?ðºw?úÎp?Äh?p`_?©U?¼hK?·A?dÉ6?",?z®"??IÅ?eá?Ìü>_­ì>¬fÝ>ºöÎ>pYÁ>ô´>+~¨> 1>Q>î­>Ì~>qm>Û:]>×N>õ?>¼Å2>­y&>}>R>Â\>*ú=Êßè=åÄØ=ÅÉ=jÍ»=îË®=ø¯¢=êi=+ë=&=øt=æ-c=×kS="ÁD=7=f*==â=T	=$ÿ<¦Úí<4XÝ<ûúÍ<¬®¿<Z`²<kþ¥<{x<H¿<Ä<öx<å­g<`W<® H<ù²:<Ì¼-<÷¬!<s<<*I<¬Ö/>Þw?D¯¡>¿«Ýû>=H>6ù¿¬Ì=?WT7¿eâ?¨½½Ø&¿÷Zz?y»8¿¹½ ¾ù?n>$¾z¿õ$>üý?õE>P¿ÿ/_¿~»«NT?&òr?SÇ>XCÃ¾éAg¿Ûx¿;#¿³È½Fà>)ØR?~?s?ñ>?kß>ÙB¾=ýÿs¾6
¿Ü>¿Z*e¿ÓGz¿ñ¿cÍx¿7±g¿QO¿v2¿ü¿ÝKá¾2¾ÿ_;¾§~½ßQ=ÃË>Ä>À­>Ö>Yú>Ã%?ío?¿(?<3?G=?Ã©E?J6M?ãÐS?ÇY?n^?­c?ßÓf?'j?Xm?¶o?½q?é¢s?(Hu?°µv?Éòw?Ìy?Dôy?Ãz?7v{?|?#|?Æ}?Õq}?cÉ}?=~?òV~?Þ~?+Á~?àë~?Ý?æ0?¥L?­d?}y???¤¨?Y´?|¾?DÇ?ÞÎ?tÕ?(Û?à?_ä?è?Hë?î?vð?ò?Yô?éõ?C÷?oø?sù?Sú?û?¿û?Qü?Ïü?=ý?ý?îý?QÆ¾¬ ?«ôZ?B.¿E¾T`?bý¿D5y?ëèx¿sï?n\l¿¼H
?¤ÆV>.gj¿$TF?wf±>îì|¿8?¢½Ù9~?ò>pÚ_¿û?¿¶~>0&{?X2?T¾ç¹Y¿D+x¿¿ð=±1?z?Ä p?å'$?W>Çr¾UÀ¿³ý]¿õÏ|¿;Û{¿óJa¿¯4¿I>ú¾÷/¾á­¼ÆáI>²È>m?{02?èM?}¶b?Oq?z??7á?¹}?¸Öx?6r?.j?Q%a?£oW?RM?LC?Á³8?á.?$?Ò?y? ?]æÿ>Ùï>Ð/à>Ñ>ÞÓÃ>¯Ý¶>®ª>Í>>Ø>øz>D>­p>Ð+`>ÐÕP>B>³)5>[´(>>_B>Ö*>"ý=Ùì=î¯Û=Ò|Ì=>U¾=!'±=á¤=¿t=âÑ=+ë=ogw=ÿ>f=FV=CiG=9=6³,=Z· =,=÷.=	=Bñ<¾Uà<iÃÐ<ªEÂ<WÉ´<<¨<Í<°<W<ÂS|<SÏj<-Z<µVK<Ó8=<Ï0<BÜ#<÷{<Ùå<à<ä÷k?âfG?Ó¤¿­;¿´t?þëõ¾Ìh<Oj>-To¾¸<²ªÄ>pW¿DNz?ÇÓÍ¾Þ!¿i$p?q9>2¿É¯ð½!8w?fø>£b*¿8t¿°eF¾è§7?ßë|?§?DV{¾
V¿<:~¿ÑÝ8¿Q¾/z±>ÃpD?Kz?m³x?ñeJ?{ÿþ>·!>[~7¾¡#ó¾ûl5¿ïV_¿w¿^ñ¿¾ùz¿Mk¿:MT¿DÎ7¿Z¿÷Ìí¾|óª¾d1S¾<Y¬½û<p>Bp>¸Á¥>ÕÎ>é®ó>yI
?°æ?×%?B1?öO;?$D?áK?¦R?HX?=»]?¹;b?'f?.i?l?to?=[q?Ms?	þt?auv?»w?lÕx?SÊy?¦z?³V{?9ö{?x|?Dø|?`}?þ¹}?ç~?eK~?Ü~?¸~?^ä~?\
?D+?ÄG?s`?Ôu?X?`?C¦?J²?´¼?¹Å?Í?LÔ?'Ú?:ß?ã?lç?·ê?í?
ð?.ò?ô?£õ?÷?:ø?Eù?,ú?ôú?¡û?7ü?¹ü?)ý?ý?ßý?nì?Ñ?ûL#>yö¿ØÛ>ñº>sJ¿N[n?{r¿S`?ÝÞ"¿ð=Jq?'ñ¿.äþ>é«#?>d¿Úº¾ùl?nJý>Ni<¿k¹_¿aÙÎ='(m?ÇP?-Â'<ÿ~B¿E~¿ÔÝ&¿Ð»Ë?s?Éw?³5?æe>Ãé!¾³%¿T¿ªy¿Æ*~¿øÄg¿¾Ú=¿NØ¿3|¾­½rW">n]·>½Á?ïI,?XI?M__?qo?,y?î~?¥ü?4~?[Þy?%s?¿k?]Þb?ÎBY?/4O?8ìD?¨:?(]0?ûT&?}?g*?ä 
?~~?½ò>Ûöâ>¯7Ô>ìLÆ>N1¹>$Þ¬>¿K¡>Éq>G>Ä>"½s>Ec>bS>ÊE>g7>Ôî*>b*>2>Ïø	>üq >Ã#ï=ÙÞ=n4Ï= ÝÀ=D³=§==¸=7°=Û²z=Pi=K!Y=_J=<=V /=¥Û"=s=Û	=þ?=ÜGô<GSã<ÕÓ<§ÜÄ<R2·<Ázª<¥<À¡<b<÷°<Áðm<ùk]<¼N<¬¾?<Ón2<&<m<.Ê<Î<aS?#½I¹|¿µ«¼Ag?©~n¿?ÌÈº¾?/¤>Öõ¾æE?><~¿fN?å`®¼^¿ØD?5è>ÒWn¿pnÃ¾¦z^?õO-?nÝø¾à°~¿îËÀ¾I6?ü?|t&?¤ýÕ½¦$B¿®þ¿)
L¿©«¾!¥>1V4?Ç´t?Ç|?´U?Õª?)`b>x¢ô½nÙ¾òº+¿öûX¿bt¿¾|¿,Ã|¿þo¿
Y¿Y=¿u¿k ú¾\1·¾cäj¾(VÙ½VÂ%<Iò=Å^>Ð·>lÇ>©4í>ãf?mW?·#?ÖC/?Ú9?ñB?ÝJ?¯xQ?ªW?IÕ\?Tsa?ye?_ùh?l?¿ªn?÷p?ÿör?ô²t?;4v?}w?j¤x?Ôy?Îyz?Ä6{?Ú{?|h|?{ã|?N}?eª}?cú}?°?~?¸{~?·¯~?ÂÜ~?Å?%?ÓB?+\?r?"??Û£?4°?åº?(Ä?-Ì?Ó?#Ù?YÞ?Ûâ?Ãæ?%ê?í?ï?Ïñ?¶ó?[õ?Éö?ø?ù?ú?Ñú?û?ü?¢ü?ý?zý?Ðý?à??Ò&¿lg4¿äöm?¸é£¾*(¾?/_!¿½?øN¾õ§¨¾8¨^?*n¿¨>rì[? 4¿p¿`I?`£2?l¿!u¿HùÔ½ÍV?ù!g?3>±©&¿ Ï¿¢2?¿è1¾¬?Ý²i?ø|?±«E?³)À>Êý½ïù¾|J¿ wu¿Ç¿im¿½F¿?¿Jx®¾j×½Bõ=3Ê¥>ìM ?J7&?ÒD?;Ý[?¤l?¶w?ä}?rü?À¼~?ÃÑz?ªÒt?\Am?{d?[?±Q?lÌF?x<?ë52?¹"(?÷S?NÙ?ª½?0?zõ>È»å>ÕÖ>ÄÈ>Î»>Å¯>öW£>"]>>r>÷áv>9f>RV>(¤G>Øð9>)->>!>·!>ªÆ>P >Eò=¦á=òëÑ=®dÃ=XÝµ=D©=L=8=<u==þ}=al=úû[=u¹L=>=sM1=ìÿ$=·=¾ä=ñù=s~÷<ÎPæ<?TÖ<£sÇ<M¹<ë¸¬<o» <û<È0<<.q<ÄU`<ÃÂP<DB<ÖÇ4<×:(<ä<®<J<Z¿ý¼féR¿®ÖB¿ñ¡5?;Æ¼>Ír¿Û0w?kHT¿Ý»F?·°Z¿iÃz?R¶q¿ªü>Ô¯¹>íä|¿h?j5?ãÿH¿H¿^7?Q¹T?µã¾®~¿°¿ÐÜ>Ë|?jQB?(=À;*¿;Ó}¿¸~\¿¦Ñ¾
>!¬"?Im?·7?'_?%$?'Z>6Çr½¿¾!¿IR¿¨­p¿Ô~¿ø(~¿5r¿È]¿N³B¿õ#¿í!¿ÈPÃ¾É:¾þ¾;¯*¼è(Ë=»öL>Ú¢>·OÀ>ñ­æ>#~?;Â?ÉJ!?î?-?þÈ7?A?Î*I?ZGP?ó~V?ì[?¨`?nÈd?¬_h?Zk?6n?²p?br?çft?=òu?>Iw?Ärx?Çty?|Tz?j{?¾{?0P|?mÎ|?Ð;}?}?²ì}?Ó3~?qq~?Ñ¦~?Õ~?ý~?Æ?Ñ=?ÕW?]n?á?Ç?j¡?®?¹?Â?ÎÊ?ïÑ?Ø?tÝ?â?æ?é?ì?-ï?nñ?bó?õ?ö?Ï÷?èø?Ûù?®ú?eû?ü?ü?ý?iý?Âý??jï¤¾øÿ¿HBÃ<¶vn?¢¢Z¿RåÀ>Q÷7¼ßô½mu¿<¿¢>Ü£5¿=Ü~?Gî8¿µµT¾Õ{?ÿ~æ¾:R¿H ?Û[?ìí°¾Mp¿7ö¾8?ùôv?jS«>â¿}Æ{¿Q"T¿ë¾Ï>i$]? ?qîS?Õ|è>Æ8;uM×¾¥>¿Ø;p¿õÿ¿r¿ÁN¿öL¿­Ä¾¢¾%¥=aþ>[hñ>ú?
²??õ0X?ij?v? }?à?-?Þ°{?v?6µn?,f?dË\?öÞR?Ö¥H?ÛQ>?
4?½ì)?g ?»?nX?@?¡cø>~è>ÆqÙ>Ú:Ë>*Õ½>{:±>nc¥>áG>8ß> >*z>ªûh>NY>13J>T<>!c/>Q#>->i>Î>'gõ=Upä=]£Ô=HìÅ=\8¸=v«==Ö=::=Ê¤=ro=£Ö^=aO=tþ@=3=1$'=ú=¿=ã³=µú<SNé<¨Ù<
Ê<F¼<÷®<¿Ñ¢<6<ÿ<°5<3t<?c<ÉxS<]ÊD<Ù 7<!j*<[<×<ÿS	<ü[¿nZr¿¢:bí?¬<º¾Þ+¿ý"m?Þû¿C+~?î¿A¸u?e4¿	Á=1?}jz¿ÀðH>Rd? `¿Þ¡N¿«j?£:p?;½|Õs¿Hä1¿¿á>
>q?Y?C9>M¿pÀw¿dj¿.÷ ¿\ ]=´?d?Ðÿ?ïCh?¢Ù*?^ç°>
-:[¤¾êô¿¿J¿Àzl¿7}¿*¿u¿ Âa¿ÁÛG¿:¬)¿z	¿¼OÏ¾Æð¾|¾0ý¼.¤=\;>f>Tû¸>à>\?7'?Yü?ç6+?nþ5?Þx??ãÉG?O?$qU? [??Û_?ªd?Äg?øj?üÀm?,p?©Fr?ät?i¯u?Ew?{@x?-Iy?¯.z?¥õz?¢{?7|?¹|?U)}?}?ÓÞ}?Ï'~?	g~?Í~?=Í~?Tö~?ë?¾8?pS?j?~?ì?ñ?ó«?6·?÷À?jÉ?»Ð?×?Ü?Má?kå?ûè?ì?½î?ñ?ó?Êô?Kö?÷?¸ø?²ù?ú?Fû?èû?tü?îü?Wý?³ý?¼bã¾Es¿u¬%¿çÚ<? Þ>Ê}¿96\?§Ï¿àPÛ>´nõ¾t2?+Ir¿íhr?6ZÎ¾ù¿i}?È¾ss¿Û:³>Äeu?Ænì½üó|¿X ü¾yô?þ~?¿vø>¢ÔÇ¾r¿·Me¿B½¾ÑH>×N?º?µ[`?÷?:¹©=.å³¾%E2¿Ðüi¿ä¿»v¿v^V¿ü%¿þQÙ¾>3A¾VÝ)=
 >íá>ó?Å:?/[T?Ñ[g?^t?3|?&©??{|?"*w?p?Ág?¥^?ê§T?fxJ?&@?¶Ù5?ü²+?ÄÉ!?¨/?)ñ?©?Jû>(?ë>0Ü>¯¯Í>_%À>Ag³>%n§>2>Tª>@Î>º)}>êk>¤Í[>äÁL>ë¶>>ô1>îd%>~ >b>²|> ø=åZç=°Z×=ÎsÈ=Pº=y§­=¯¡=ml=1ÿ=pJ=	r=E±a=	R=dwC=¡ç5=rH)=:=}=Óm=ëý<ÖKì<åÛ<¡Ì<>m¾<;5±<è¤<pu<7Î<Iä<Uw<Z)f<Î.V<6PG<Üy9<k,<Ñ <,w<´<Z^e¿«G¾~)C?%Ô,?«f¿XÇ>D?5Q¿"Sg?«`¿7?»T¥¾!¤¾Ij?äéV¿w;¾Ã}?U¾fÍo¿Ò>ÐI~?§><Ï^¿Q¿µ=òÕ_?[²k?d£>Ò¥ã¾4Þm¿jt¿¦ð¿ºl<½Tö>-Y??­o?9º7?­µÏ>»õ{=r¾{ï¿ÓåB¿Ëg¿üf{¿Ç¿Nw¿{ºe¿
ÑL¿T=/¿´ø¿9,Û¾y¾Ê/¾²ÍR½¤Iz=Ø()>Z>·±>y|Ù>\5ý>z?|¨?Ð()?5/4?òá=?"eF?QÚM?C`T?òZ?_?Ã`c?¥&g?oj?ðIm?Åo?Öìq?êËs?¾ku?Ôv?x?y?fz?vÔz?X{?¥|?£|?}?\z}?ÇÐ}?¤~?~\~?«~?TÅ~?zï~?ü?3?üN?´f?>{??p?È©?Vµ?W¿?È?Ï?Ö?£Û?à?»ä?cè?ë?Kî?ªð?¸ò?ô?ö?`÷?ø?ù?fú?'û?Íû?]ü?Úü?Fý?£ý?}o~¿¢P¿¡E%>Åv?6î¾Á:)¿»T~?Ëdp¿OX?Z¿Õs?ï~¿º|;?·¼¬N¿Ñb?ä­(>ºò¿}g«=,Ú?ãø=	3n¿Ù(¿îÐ>8ü~?h?Rz¾id¿Çfr¿a°ö¾ä=>Ã<?F}?ô×j?|Ï?W»&>z¾%¿EÁb¿~¿Ó-z¿ e]¿`F/¿î¾
lk¾$Û;¨ª_>â/Ò>}?!b5?¦\P?Ã}d?í}r?!+{?V?Æ?ã1}?ë;x?Zqq?hJi?Å+`?|hV?DL?ôA?¦¤7?mu-?#?×?Õ?f?Ì.þ>ýí>Ò¤Þ>#Ð>itÂ>µ>x©>>ðt>{>R&>ÿØn>^>@PO>A>Ö3>x'>«ï>/>À*>ó©û=VEê=éÚ=@ûÊ=4î¼=ÒØ¯=Qª£=úR=!Ä=ð=öu=ßd=±T=PðE=³48=±l+=w=Zu=Â'
= =VIï<t­Þ<8Ï<6ÖÀ<bs³<]þ¦<ªf<î<â<rvz<%i<ÓäX<ÖI<ßÒ;<µÈ.<G¦"<[<iÙ<µâ½èe?Ç¤|?-w½*xt¿>@?zé½Q°¾<	?G¿Yi>0þ=ÇO.¿ï?¶¿¶eí¾½|?ß¤¼¿ð%=d~?Ù»>l@¿·i¿¾+¶½BPH?«;x?n4ç>Ç¦¤¾rS`¿I{¿a)-¿öý¾ÏË>X¤L?n|?3¾u?ÝµC?Õ£í>J3ù= Y¾÷ ¿E:¿¢b¿$y¿Öÿ¿Ñy¿oi¿æQ¿§4¿}»¿Häæ¾5¤¾F¾`½Ï,=¢,>Nz>R1ª>jÒÒ>z@÷>à?FO?¹'?_[2?ÐF<?üD?§L?TLS?"Y?9^?½©b?Rf?äi?sÑl?h\o?êq?ú|s?<'u? v?Úw?Pðx?£áy?Ü²z?8h{?h|? |?ª}?ñi}?Â}?Q~?ÑQ~?l~?R½~?è~?ù?f.?zJ?Ìb?Ýw??ç?§?o³?²½?Æ?FÎ?ðÔ?¶Ú?µß?	ä?Éç?ë?×í?Fð?bò?5ô?Êõ?(÷?Xø?^ù?Bú?û?²û?Eü?Åü?4ý?ý?W@!¿K³¼É6[?çö$?5^¿IÉÇ¼ê8?qÔw¿ÿ?²ì¿-|?{V¿4nÇ>ñ&¹>©þv¿H.?È4í>°pv¿7qA¾7Èy?õ­³>£äS¿¦L¿²h>¤ïv?é9>?§¾½ÀQ¿÷1{¿})¿"`=Óã(?³Jx?öKs?êî*?
w>íT¾Lú¿Z¿²À{¿Õ|¿Ðc¿T&8¿1¿×¾½ã;> 4Â>eS?Hý/?6L?¹ya?]zp?Úz?yç~?Èî?¯Ó}?M<y?z¹r?÷Æj?±Ìa? X?¼N?T½C?âj9?4/?$3%?ä{?l?s	?$ ?½¹ð>§;á>Ò>EÂÄ>ø½·>I«>q >?>Z(>s·>ßÆq>Ga>CÞQ>å{C>ï6>)>³Þ>ôü>µØ>Ëþ=§/í=	ÉÜ=Í=I¿=
²=è´¥=9==«=Ù¤x=rfg=YW=7iH=Á:=ì-=²!=5P=¯á=\,=ÕFò<Øuá<ÏÑ<,?Ã<±µ<«©<ãW<¥k<{A<Ü}<îük<Ø[<å[L<á+><þ÷0<½®$<Õ?<<äÔF?Mð?p7?ÃC¿<þ¾ì?t	1¿_Q>wØa;2ÒÆ¼[¾1¾ËÂ?
Èk¿Â¬n?º¾v;¿UÞb?\>d{¿aQ`¾Ü·o?+¨?ùò¿zMy¿È¾Q+?å~?	?vE¾|UO¿¯Z¿±b@¿£cy¾;H>>? hx?Jmz? ½N?ÄH?iÁ9> ¾½é¾xÕ1¿[]¿pv¿%Ó¿±{¿¢Þl¿V¿Ëé9¿¸a¿õuò¾Q¯¾^ \¾HJ½½ÇÇ»</$>×i>¼¢>EÌ>V@ñ>?4	?Ëð?±ý$?ø0?§:?:C?_K?Y5R?p.X?e]?ða?!æe?EXi?Wl?uòn?ä5q?-s?åát?ö\v?Ò¥w?Ãx?eºy?×z?»J{?Ùë{?{w|?zð|?PY}?%´}?Ö~?G~?~?5µ~?á~?ä? )?êE?Ø^?pt?$?V?\¥?±?¼?#Å?Í?ÛÓ?ÆÙ?åÞ?Uã?-ç?ê?bí?áï?
ò?éó?õ?ïö?&ø?4ù?ú?çú?û?-ü?°ü?"ý?ý?_ >íçI?ú½r?å×½0>y¿j©?æ >U|'¿#Y?ÏÇ`¿wJ?¥¿â°¼nÐ0?$¾~¿]¡Ë>JF7?´¾W¿áå¾Cc?ò£?gO/¿ß¡g¿J8=Eg?0ÙW?¡Ïy=8j:¿O¿«.¿mñ5½H?¸Óp?¥y?hÆ:?½Y£>ûÛ¾ê1¿çtQ¿}x¿Ã°~¿oi¿@¿¿D¾þ½E'>öþ±>b|?lq*?`èG?5P^?.Tn?h½x?V]~?ÿ?î`~?7+z?Þòs?-7l?Vcc?4ÐY?aÆO?QE?`,;?¼î0?ã&?%?é®?É
?~÷?«só>¨Ðã>xÕ>ïÇ>âç¹>°­>¶ì¡>¥>ÉÔ>?H>5´t>d>íkT>öÝE>I8>Þ+>Í>;Ê>
>ö >Øð=ß=æ	Ð=É£Á=];´=t¿§=û=íM=@;=´µ{=ý@j=Z=âJ=ËÎ<=$µ/=ê#=+==¡Ç=RDõ<:>ä<zfÔ<!¨Å<­ï·<ø*«<I<[:<ð<£\<¸æn<ÝP^<½áN<ã@<G'3<3·&<)$<Ó^<s?b?¾¡¢¾Ö~¿s·i>:H?ïÑ|¿ùA?t¾¿Dõ>*?¿[?¹ð¿!9?¹Ê=Ûáj¿T¼2?þÍ	?òd¿¤wê¾çS?Ê:?xÚ¾QÐ¿YDÜ¾~¤	? ?:u/?öwx½7';¿P¿¿ÑcQ¿µ¥­¾,Áb>/?r?	´}?ÃÃX?È/?;v>EâÎ½ÚeÑ¾»¨(¿îV¿Ls¿A¿;}¿1p¿qZ¿Ë?¿Qê¿Tßý¾'íº¾Å#r¾ç½	z;ë!æ==PY>ü=>f]Å>75ë>ø? ?Éà"?¦.?9?! B?&J?XQ?#8W?(\?W5a?Ce?fÊh?*Ük?@n?ÄØp?9Ür?·t? v?ÿpw?>x?¬y?gnz?â,{?ûÑ{?a|?Ý|?|H}?¥}?4ö}?<~?x~?ÿ¬~?gÚ~?»?Ë#?KA?ØZ?ùp?#?¼?£?¯?Vº?¬Ã?ÂË?ÂÒ?ÓØ?Þ?â?æ?øé?ìì?zï?±ñ?ó?Eõ?µö?ô÷?ù?÷ù?Æú?zû?ü?ü?ý?uý?¼æw?Ôv?«¾>lWJ¿-¿Ðëz?ß®ñ¾ÕÞ½ØÞ>Xy¿åÑ>¤â±½©Û¾þ+j?dåd¿D=SÅe?Z&¿_0,¿l­>?¾f=?9¿Kþx¿Ú¾ûøO?¿k?Á|Z>`r¿S¿ö}D¿¬#¾½ò÷>¼ôf?Õ}?7I?áÉ>nÈq½ ñ¾ÀvG¿?Qt¿É½¿Än¿ÌH¿ª¿Ì£³¾Óüë½Î$â=/¡>eý>È¿$?DtC?»[?±l?ôWw?½·}?Vø?Ù~?{?su?ùm? ïd?7w[?ï|Q?w=G?é<?¥2?Õ(?É½?F??f?re?R+ö>Òcæ>ot×>dZÉ>Ñ¼>L¯>WÔ£>ºÑ>Ï>´Ø>¡w>¸¾f>=ùV>¾?H>:>{°->P¼!>c>U4>>êó=û6â=Ò=zþÃ=l¶=õÉ©=o=É=Ðà=Æ~=m=©\=õZM=Ð?=ZÙ1= %=å=U=åb=ÌAø<ç<nýÖ<È<Ò-º<DA­<S:¡<	<«<Xí<Ðq<áa<gQ<åÝB<V5<©¿(<}<!<r>^Î¾/m¿Ó¿i¡T?öJ>"¯a¿é{~?>f¿_sZ?äi¿R?¡Ag¿àÎ>EHå> c¿dÁá>×tB?Ep=¿èÐ*¿Ú:,?/g\?eïm¾÷}¿¹F¿iÈ>Úz?HH?è©=$¿×»|¿ú_¿mÝÜ¾û>ÁC?kBk?¹?Ç»a?õw ?Ðç>ã8½5¬¸¾¿½hP¿Ã¹o¿xK~¿<o~¿Ìêr¿ ^¿sñC¿8T%¿@¿-Æ¾~¾ÿm¾áz¼ÕèÁ= ¹H>ôµ>'¾>cå>bÌ?Z$?¿ ?¨Ä,?\7?M¬@?X×H?UþO?'?V?è´[?úw`?+d?æ:h?__k?Ên?zp?ir?³Tt?râu?;w?âfx?yjy?Kz?¬{?Ì·{?_J|?cÉ|?r7}?Î}?ké}?ü0~?ün~?°¤~?4Ó~?~û~?d?<?ËV?wm???Ô ?­?¡¸?1Â?zÊ?¦Ñ?Ý×?>Ý?æá?ïå?mé?tì?ï?Wñ?Nó?õ?{ö?Â÷?Ýø?Òù?¦ú?]û?üû?ü?ýü?eý?\²;?Bý¹>jî¾÷A}¿íd">lÞ`?·h¿"gó>¹|æ½ì.Î¼$½ÐS·>ÖB¿Üí?Ý,¿×Þ¾{}?äâÍ¾Ù Y¿Øu?¯`?©¥¾)ä¿ä±¬¾I2?Üx?í¬¸>À5¿Jz¿¹JW¿|9¾yÆ>þÆZ?¸Ò?®&V?îï>ðu<{nÑ¾W¡<¿Ao¿Õû¿DEs¿ZP¿_ô¿D±Ç¾Ï¾à­=¤û>CÓî>é?£Ú>?ÚW?9¡i?«Óu?Áö|?8Ù?=?]Ô{?'9v?Hòn?~qf?]?V,S?¹ôH?î >?ZX4?W9*?ÊZ ?|Í?C?úÑ?«àø>õè>äáÙ>¡¤Ë>Ä8¾>±>S»¥>J>h,>Óh>@z>åyi>2Y>=¡J>²º<>èÂ/>åª#>md>ÿá>Ø>Úíõ=Ííä=8Õ=YÆ=±¸=kÔ«=Ùì=×=Y=¦ë=þõo=lQ_=ÎÓO=ÒhA=ý3=S'=¹à=l=(þ=D?û<øÎé<`Ù<zÊ<õk¼<W¯<+£<Å×<CM<~<Iºt<å¼c<jíS<æ6E<Ù7<È*<Ñì<<ä<.¿n¿;b¿Q{>¡Â|?5´ô¾úYÕ¾º8a?¥_~¿=ë?pw¿Öo?ì&¿»À<ËÒ<?è®v¿M>Ücj?Ð¿§]U¿ÆÔõ>s?Âî¼ÿp¿ ¶7¿ p>bÄn?àÿ\?vQ>é
¿Zv¿¾ûk¿ð¿e==ÊD?ß^b?°÷?i?÷-?ï"¶>þÆ8<!q¾õ!¿çuI¿Ù¹k¿ñ|¿L¿u¿qb¿«´H¿a*¿Ì
¿|PÑ¾¤æ¾u=¾Fä½ =ð8>ô$>ã¾·>!ÿÞ>?¶??ØÞ*?é°5?Ç4??4G?RÞN?CU?JÙZ?¸_?i÷c?Å©g?&áj?­m?<p?¥7r?Út?¤u?tw?ø7x?ËAy?H(z?ðz?L{?h3|?}µ|?5&}?Þ}?zÜ}?Ç%~?Fe~?F~?ëË~?/õ~?í?â7?±R?êi?~?q??«?æ¶?²À?.É?Ð?ãÖ?fÜ?,á?Må?áè?ûë?ªî?üð? ó?¾ô?@ö?÷?°ø?«ù?ú?Aû?ãû?qü?ëü?Tý?+M4¾¹¿ÕÍy¿!l¿æúI?°^»>Úy¿d?ÿ¿Çô>$¿C;?¶¼u¿EÇn?W¼¾è¾¿¢|?ÃFô½k¥u¿6¦>ú&w?¡Ë¼½C	|¿«¿ú?ÛY?¼ÿ>¦À¾dfq¿ÚÇf¿`Ã¾µ>§iL?´?ðza?b	?Að¹=]X°¾^ 1¿PQi¿´j¿w¿W¿,ë&¿ÖcÛ¾xaE¾W=>7>ýbà>;ð?_:?#øS?g?½0t?v|?/¢?÷?||?èEw?=p?ßèg?9«^?ÔT?
¦J?êS@?/6?ß+? õ!?Y?\?=?²û>ë>ÑMÜ> íÍ>¶_À>³>¨¡§>Sb>×>ø>ñx}>4l>Ê\>qM>'ó>>$Õ1>R%>V1>>§>©×ø=¤ç=A×=©³È=ÆÎº=ÖÞ­=;Ó¡=i=Ý+=t=sÐr=Sùa=¡LR=ÐµC=º!6=~)=»=SÉ=i=º<þ<Uì<Q+Ü<ùâÌ<ª¾<Ûm±<Á¥<z¦<Úû<Á<¤w<èrf<AsV<èG<"µ9<Ð,<%Ñ <ð¦< |¿2Y¿:í_¾4P?*J?ã=n¿ß_>_è>a×G¿ûã`?e
Z¿;Å.?@¾ÿ¸¾°n?ÄQ¿¹3¾',~?j#¾"r¿nq>ðè~?ò3>ü5\¿ùYT¿½=KÇ]?Ù/m?woª>ÆÝ¾+±l¿ Du¿t¿¯Äd½pqò>4X?Wñ~? Vp?2ó8?×³Ò>ÿ(=MÆ¾ÇÓ
¿-B¿£Ng¿3{¿¦Ñ¿Ò×w¿f¿fKM¿ËÇ/¿d¿ÀUÜ¾û¶¾òû1¾h[½r=Yc'>s>öà°>¸ÔØ>{ü>ÓC?mm?¨ô(?E4?¹=?¾AF?V»M?/ET?NûY?÷ö^?ÏNc?g?aj?>m?Ôºo?ëãq?,Äs?eu?»Îv?x?£y?z?*Ñz?}{?,|?[¡|?Â}?Àx}?bÏ}?n~?r[~?Ã~?Ä~?Ìî~?e?3?N?Rf?éz?¿?0?©?&µ?.¿?ÞÇ?dÏ?çÕ?Û?nà?ªä?Tè?ë??î? ð?°ò?yô?ö?[÷?ø?ù?cú?#û?Êû?[ü?Øü?Dý?Ægl¿é&¿ì}L¿Ä>>ô ?¢¾ûG%¿c»}?¹èq¿¡TZ?É\¿û±t?´}¿¶S9?cm3¼ÓEP¿5¡a?/ÿ1>¤û¿-=ìè?¸>m¿©ÿ)¿*RÎ>Ý~?d_ ?(Ûu¾,íc¿\¸r¿Pø¾ß:>s<?w*}?k?ÏS?¼)>~f¾a $¿b¿Ü
~¿Dz¿T]¿>/¿²î¾¦l¾p:ä^>H¹Ñ>ñÔ?a:5?0>P?Âgd?^or?ô"{?AS?£Ç?ç6}?§Cx?${q?±Ui?8`?uuV?^QL?øB?ú±7?-?Å#?`ã?¬?¹¦?^Dþ>	î>3¸Þ>`5Ð>¦Â>N£µ>T©>Ó)>R>>	2>åîn>^>YcO>^+A>/ç3>'> þ>=>N7>WÁû= [ê=5&Ú='Ë=Íÿ¼=6é¯=¹£=/a=[Ñ=`ü=àªu=3¡d=oÅT=ÊF=æE8=²|+=\=7=©4
= =°_ï<@ÂÞ<éKÏ<9èÀ<%³<÷§<.u<qª<t<Øz<ë(i<ùX<éèI<kä;<	Ù.<yµ"<¥i<tÄ¾ä¦½?¸{?¾´½n¿r¿~C?8¾ë§¾«?>ì¿x>q>M0¿ü?qÛ¿Ýâñ¾Q|? =¼ÖC¿|`Ù<ßä}?¦¾>h?¿eKj¿MÂÀ½G?x?0é>DÃ¢¾«à_¿å·{¿À-¿ÿò¾)Ê> @L?%||?Bæu?ÕD?Õî>NÉü=m{W¾V1 ¿nV:¿$zb¿çy¿öÿ¿Éày¿¹i¿ ´Q¿vÏ4¿æ¿M;ç¾'t¤¾8§F¾½NÖ)=¦>ËÓy>½ù©>o Ò>É÷>;Ì?¤=?$'?¥M2?¿:<?ýñD?cL?9DS?øY?U3^?]¤b?¤f?oài?éÍl?UYo?>q?¨zs?9%u?av?Øw? ïx?ày?ß±z?]g{?ª|?ü|?}?ui}?"Â}?ô~?Q~?&~?½~?Vè~?Ì?>.?XJ?¯b?Ãw??Ô?§?`³?¥½?Æ?=Î?èÔ?¯Ú?¯ß?ä?Äç?ë?Ôí?Cð?_ò?3ô?Èõ?&÷?Vø?]ù?Aú?û?±û?Eü?Åü?4ý?§bR¿@)¿öÜr½»yV?59+?
FZ¿)ÍV½Iz=?ùPy¿»é?±¼¿¥ï|?ÀþX¿áfÏ>Í±>·úu¿£Ô0?jç>Q>w¿%Z6¾
Xz?è¯>)U¿CBK¿ìZp>®iw?×=?å¿Ê½àgR¿=íz¿ ¿gÝ=Z¸)?fx?2ÿr?®D*?Lt>YW¾¼¿úéZ¿iÝ{¿2¾|¿c¿Í7¿NÊ ¿¸Æ¾(½~<>ãÙÂ>?50?aL?a?Æp?Uz?tì~?í?Í}?S2y?¬r?ä·j?¼a?X?¨öM?«C?³X9?!"/?²!%?k?/?é	?Uy ?ð>!á>Ü{Ò>ªÄ>ª§·>Ul«>Éð> ,>>Q§>¶¨q>ß*a>õÃQ>XcC>ù5>¶u)>ÉÊ>dê>iÇ>ãªþ=¡í=­Ü=hÍ=Æ0¿=ó±=â¥=ì%=Òv=µ=Ex=Ig=8>W=ÀOH=j:=Þz-=+q!===çÏ=Ð=	(ò<.Yá<Ù´Ñ<Y&Ã<oµ<,ÿ¨<áC<Y<(0<w}<íÞk<ì~[<éAL<³><~á0<Í$<Y,<(Û?X/@?³?¬Å?ÉO>¿mÃ¿Ò¥?Z%,¿vh>Õ< 9½ô¾ÙÓ?Ij¿Kp?äØ¾9¿é[d?Ä>è{¿V¾}p?!Å?:¢¿Ùx¿\¾À,?¾~? Ó?ÝÙJ¾¿P¿]D¿N§?¿6du¾Q¡>®*??¦x?÷Cz?èPN?w¶?N7>:Ó"¾-ê¾¾12¿>]¿¡v¿äÖ¿â{¿¼l¿\ïU¿k´9¿(¿ÿñ¾Ô¯¾=[¾ö»½­Â<"Þ>j>	£>bÌ>[~ñ>ÞO	?K	?Z%?0?L¸:?õC?~lK?¢@R?K8X?m]?øa?¦ìe?ñ]i?u\l?¾ön?9q?P0s?³ät?e_v?ï§w?ãÄx?ü»y?7z?ìK{?âì{?`x|?@ñ|?ýY}?»´}?X~?rG~?p~?µ~?Ìá~?#?W)?F?_?t?B?p?s¥?±?¼?2Å?Í?æÓ?ÏÙ?îÞ?\ã?3ç?ê?gí?åï?ò?ìó?õ?ñö?(ø?5ù?ú?èú?û?.ü?±ü?#ý?Ðý=NdT>¤Ò8?Ö[y?õB¼93}¿jå?Óqh>Kð3¿ a?èog¿dR?Ùc¿ èÉ<³¯(?¿ýÜ>­÷0?,\¿å;×¾Äf?ý
?ÙÅ3¿ze¿qóq=Ë3i?²PU?½4=&&=¿/F¿y,¿¡½"½?¿Ãq?y?¥9?b>$3¾Ù	¿ÞR¿äx¿²~¿¤i¿=¯?¿· 
¿¾4½_5>È³>>?þ+?cH?Ëª^?2n?¹âx?Óm~?Ûþ?rR~?ßz?9Ñs?gl??7c?K¡Y?ÛO?%OE?Qû:?`¾0?à³&?hð?à?u
?GÏ?>'ó>Cã>ÁÔ>qÎÆ>0«¹>¨P­>2·¡>}Ö>Ó¥>O>bt>[¶c>C$T>E>¯
8>«c+>Q>§>oW
>&Ê >Èï=Ù3ß=ìÂÏ=°aÁ=Òý³='§=£ê=D==£_{=áði=ü¶Y=²J=3<=y/=÷K#=üö=$k==`ðô<ðã<ÇÔ<ydÅ<·°·<aðª<<<ÛÀ<²0<ïn<Â^<êN<ûB@<óé2< ~&<ï<íÖ?Ïnz?["1?H»g¾Aö¿Õ>»S?8Qy¿÷6?>8ô¾P×Ú>n=¿¿T?ì¿g@?òq=
íf¿û8?£Å?Hh¿¨Ý¾KW?ßA6?Ácä¾¿g6Ó¾`?+À?È,?½B=¿íÞ¿Ë£O¿éZ¨¾Êîl>×0?lUs?Ði}?T·W?ý°?¤o>ÏqÛ½XÔ¾e¯)¿FW¿+ªs¿}V¿}¿Ý²o¿­úY¿»u>¿tP¿ÿ ü¾¬¯¹¾6»o¾\â½ ¬Á;ºê=![>Ù>dÆ>gßë>ÑÎ?tÐ?V#?Ú.?C29?­HB?¬@J?k:Q?GSW?Ö¥\?úIa?Ue?	Úh?Ãék?n?ãp?#år?s£t?È&v?Òvw?Kx?y?3rz?,0{?ÕÔ{?c|?1ß|?WJ}?,§}?÷}?E=~? y~?ç­~?0Û~?i?a$?ÍA?I[?[q?x??[£?Æ¯?º?ÖÃ?æË?áÒ?íØ?*Þ?³â? æ?ê?ùì?ï?»ñ?¥ó?Mõ?¼ö?ú÷?ù?üù?Êú?}û?ü?ü?ý?-\?h?®¬~?gã?¯1¿Ü¿)¿·£r?­»¾\¾ÖÝ?±T¿ÄÅù>Y+¾ý¸¾Tb?æk¿Ï>f_?t0¿ú±"¿EF?ë5?ÇÔ
¿L£v¿,ò½ÍT?"h?aÑ>>$¿²¿
Ì@¿ó¾çB ?åh?(>}?½F?äÃ>üU½%÷¾QI¿d!u¿Î¿Päm¿®,G¿Oø¿°¾U¯Ý½Vsï=S¤>¬ÿ>Ç%?;CD?[?àvl?>w?i×}?bû?}Å~?<âz?ét?,\m?k©d?,[?ë.Q?-îF?É<?;W2?IC(?s?»÷?ÓÚ?$?ç®õ>éíå>û×>FñÈ>Þ­»>M4¯>}£>é>/4>>ñw>uAf>DV>ÒG>#:>xQ->¹c!>ÏD>_ç>Ê>>O~ò=ºá=3Ò=Ã=¶=cl©=Q¯=¯Á=Q=ø9~=®l=º/\=éL=T²>=,w1=Á&%=Û°=`=>=¶¸÷<æ<´Ö<¢Ç<ÿÆ¹<á¬<Gá <3¶<Q<¥<ñJq<`<êóP<CrB<gò4<sb(<Á±<ö?,
Ö>Þ"Ð½z\¿TK8¿¨ ??1>£>u.n¿ýy?Ø5Z¿i M?§x_¿ûc|?¶án¿í@ï>çßÆ>xÖ}¿gû>mj9?á¦E¿sð!¿¤4?øW?s2¾«2~¿¿ðHÖ>{?ÇD?¦°G=k(¿Í}¿)]¿ÃÔ¾bâ>¶]!?±l?ßS?ö4`?Z#?É>»Fa½Ë#½¾ÛÓ ¿ØQ¿ªep¿î~~¿>~¿Çkr¿¬Õ]¿{C¿R^$¿¿_+Ä¾¿¾J²¾B¼ÆfÈ=6³K>ê>3Ë¿>'7æ>)I?.?&!!?2-?¬¨7?*ï@?ðI?1P?ñkV?ýÛ[?`?×»d?¸Th?Óuk?N.n?p?"r?{at?íu?)Ew?:ox?µqy?ÓQz?{?¼{?tN|?íÌ|?:}?v}?¸ë}?û2~?¶p~?/¦~?Ô~?ü~?]?u=?W?n?¥??=¡?ð­?ð¸?vÂ?µÊ?ÙÑ?	Ø?dÝ?â?æ?é?ì?%ï?hñ?\ó?õ?ö?Ë÷?äø?Øù?¬ú?bû?ü?ü?ý?÷Üd?é¿`?û/?¤'¾O¡¿ÆúÝ¼±r?ßT¿SE­>?èÜ<ÄN¾g=Xq>f0¿l ~?µC=¿¦=¾Éây?ï¾ðCO¿®?îX?t¸¾p-¿_¾ª::?ê.v?E¦>,ÿ¿?/|¿)ëR¿z¾2Ó>^^?á?CS?µüå>î1»Ù¾f?¿Fp¿Ãÿ¿6r¿pAN¿­¿¦½Â¾5ï¾&5ª=ê>`ò>H` ?Á@?¡mX?>j?7v?D)}?/ã?­&?^£{?ôu?"n?f?j¯\?ÌÁR?H?4>?©ì3?æÏ)?fô?¹j?>?w?4ø>òQè>GÙ>Ë>±¯½>A±>]B¥>à(>.Â>l>ZÓy>.Ìh>õãX>Ì	J>d-<>?/>ÿ/#>Üñ>:w>\³>|4õ=#Aä=iwÔ=YÃÅ=?¸=R«=øs=g==#=u@o=t¨^=6O=rÖ@=Pu3='=¹j=¡=ó=	ú<ïé<ïØ<µàÉ<FÝ»<ÈÒ®<ù¯¢<Èd<@â<x<ò t<kc<êLS<¡D<Üú6<ÇF*<tt<èiå¾>$õ¾íØR¿@Èv¿©\½ðç?ám£¾Ì®¿åp?+è¿ì|?¿MÐw?V9¿ëN÷=Ya,?{¿ïu^>ì=b?´A¿JüK¿±í?Eàn?Ü	¤½YÓt¿³§/¿Ôl> r?DBX?Q"0>|¿í?x¿Õ?i¿¾ëþ¾¬Ùv=4Ø?¸d?Âÿ?¢Àg?í*?î®>:¿0»qÁ¥¾É£¿:K¿gÂl¿P}¿F¿¢æt¿~a¿ÉG¿ÞP)¿º¿Î¾Ù3¾ÿ¾:æò¼Æ¥¦=
9<>%>Fr¹>Óà>þ¾?Q?Õ!?ýW+?6?r??OàG?-&O?IU?[?Kè_?!d?ýÍg?§ k?vÈm?3p?MLr?Ét?©³u?ôw?®Cx?óKy?1z?»÷z?é£{?$9|?uº|?*}?}?µß}?(~?²g~?`~?¼Í~?Âö~?J?9?·S?Íj?Ë~???¬?U·?Á?É?ÎÐ?"×?Ü?Zá?vå?é?ì?Äî?ñ?ó?Ïô?Oö?÷?»ø?µù?ú?Hû?êû?vü?ïü?ûÙ=êÙ>ò¾-a¿«oD¿o?Ð?`ú¿Z»I?§Ðó¾Á¦>øÓÅ¾ ?ði¿×hx?µ÷ñ¾Sö¾dF?ÒX¾ärn¿ÜÌ>lq?¥%¾Qm~¿éßé¾ø¿?#
~?Ç¸é>ùÖ¾Ët¿;b¿I±¾Ü`£>e5Q?Yé?À^?LÎ?7C=4÷º¾4È4¿pLk¿K¯¿Þõu¿ÈéT¿Ø$¿ü*Õ¾@Ô8¾KI=U> å>;Ú?[¢;? U?èg??¹t?uc|?F¶?úu?:U|?òv?;Õo?ªrg?3+^?qNT?ëJ?%Ê??£~5?°Y+?ôr!?ØÛ?´ ?ÉÉ?>¸ú>Z´ê>áÛ>Ã3Í>§°¿>ù²>§>cÑ>ÏO>y>G|>Vk>WC[>È@L>q>>>,1>#ü$>Î>ÿ>Ü'>ê÷=¥Çæ=ÑÖ=ôÇ=cº=¾8­=8¡=s=Ù¥=E÷=5èq=(!a=mQ=úB=ps5=NÜ(=$=Ó<=¨=[Iý<×´ë<XÛ<ÒÌ<ó½<ûÃ°<«~¤<]<ór<Z<ó¶v<@e<é¥U<ÒÐF<P9<+,<(7 <Þ~¿v}¿3w¿@ó¾4)$?ÏH?4ÉS¿ V<¢?/a¿
r?`l¿ZÑG?íÏ¾w¾a?è]`¿ ½C5z?=Pº¾Mj¿(Lª>Û |?î®â=G»c¿ºîK¿®÷ü=RÅc?¶h?Ñ>e¡ï¾Þp¿âr¿ ¿_×¼Äþ>v[?¤l?6Rn?A5?Ö²É>6K=|ú¾$¿ª{D¿ÐÁh¿Ë{¿°¿Î"w¿S÷d¿ÊÚK¿E'.¿Ò¿ØØ¾ÑH¾|e+¾¡;B½Ù=p³,>èö>è³>¦ËÚ>É`þ>?q?û)?ñ4?2>?Î«F?-N?TT?$BZ?»4_?c?ÙEg?>j?am?Ùo?¤þq?_Ûs?(yu?3àv?¨x?Æ%y?ýz?
Ûz?{?#|?È§|?T}?}}?Ó}?~?^~?{~?æÆ~?Öð~?)?4?ÞO?xg?è{??ï?6ª?µµ?ª¿?IÈ?ÁÏ?8Ö?ÑÛ?«à?Þä?è?§ë?aî?¾ð?Éò?ô?ö?k÷?ø?ù?nú?-û?Òû?bü?Þü?G¿vÈ5¿Îhg¿Ës¿gÞþ½`Èz?_X½L½P¿Õ²~?Z¿ãò:?N@¿¸Ëc?>ê¿×îQ?Øv¾§î:¿·n?Ìx=<;~¿	Ë1>/5~?6/=ìit¿`¿Ìî>4â?ú0?P¾¤i¿|n¿¨
ä¾A=d>÷B?Ø^~?Ig?à?´>¤¾g)¿*Be¿ «~¿Úy¿7"[¿ý=,¿KDç¾5z]¾à²y<½k>ñm×>t6?Å"7?Ý³Q?,ue?
!s?{?ªt?^³?Ä÷|?%ãw?jq?¡Éh?f_?ÑÔU?¬K?÷[A?!7? à,?.ï"?K?V?Î?à9ý>í>ÖÈÝ>eSÏ>½°Á>Û´>EË¨>py>Ý>Xí>¹B>uàm>g¢]>wN>IO@>â3>%È&>¤K>­>J> ú=Né=+Ù=Å$Ê={&¼=Ý¯=.ý¢=Ë±=.=dd=ït=Öc=NÐS=¤E=q7=·*=oÞ=
Ø=Z	=Õ =¾Kî<sÁÝ<í\Î<Ò	À<-µ²<\M¦<ñÁ<¤<<<óly<h<éþW< I<Ä;<l.<Üù!<ª^ ¿@4¿ZõÚ¾`7>~?5®M>¤¿¹3?à>Î=
¿ã.?Nû(¿Õé>Ó¼©¿P}?è.¿÷î¸¾:?øQð½|¿&êñ=ÆÃ?ÁI>ztK¿`£b¿o÷¼@¿P?¸t?üCÑ>®æ¹¾©4e¿Zy¿KW&¿Pµæ½m0Ú>;øP?:}?âs?ê×??VÔã>UuÐ=¶ºk¾Y¿äc=¿{ed¿Âðy¿cø¿¶y¿g<h¿¨P¿¸à2¿Ö¿ã¾M ¾ó§>¾6z½/F=[#>&¿>d§¬>ØÕ>ç:ù>wÁ??6Æ'?Ýö2?zÏ<?ptE?M?¨S?'rY?\^?æb?O¼f?j?ùl?:o?(°q?<s?>u?å¬v?)ëw?/ÿx?îy?	¾z?èq{?Í|?ç|?ø	}?go}?HÇ}?j~?^U~?~?ü¿~?Ùê~?ù?!0?úK?d?ýx??¿?P¨?´?>¾?Ç?°Î?LÕ?Û?úß?Dä?üç?4ë?þí?gð?ò?Nô?ßõ?;÷?hø?lù?Nú?û?»û?Mü?Ìü?Âr¿ó%}¿ôRi¿uõà¾zÏ?3a?0[%¿æ!¯¾9e?æ¿Zw?vàv¿ö¦?ün¿gÅ?É>>nh¿Ú¢I?"¨>}¿-½©~?ü{>a¿¡ê<¿f* >ð«{?Ù{0?ÕÏ'¾¢äZ¿¦w¿Dò	¿öáþ=vK2?Rìz?o?#?R>Rw¾/¿Y~^¿võ|¿¢²{¿xç`¿»4¿qù¾Hê¾¼<L>«É>v?¾2?t)N?©åb?nq?#z?_?ÕÞ?ó}?,Çx?"r?jj?÷a?ÞTW?÷6M?éB?8?±d.?i$?c¸?k`?j?u¹ÿ>6tï>sà>ðqÑ>ò¯Ã>å»¶>Ýª>!>òi>Ú`>Öü>jp>&`>û­P>ì_B>5>(>_ø>E&>§>VVý=eÔë=Û=eUÌ=0¾=ñ±=½Á¤=W=L¶=~Ñ=¡7w=f=*V=·BG=¨o9=Ò,=G =@s==ül=£âð<[*à<Ð< Â<_¦´<¨<p<V<y<ó"|<ç¡j<èWZ<`/K<8=<¿ó/<¼#<¢>Bk>¬Ò>O÷e?O?÷{ó¾^nC¿Úp?ùã¾ØDä¼Y'>w¾qU=ì·>ÏÐS¿u{?Ã3Ø¾ùº¿RÎq?I>¿éÑ½K'x?r\ò>ôÂ,¿Õ&s¿a;¾ªq9?ü|?H¿?
²¾§W¿{~¿´7¿ìJ¾C2´>´ME?Éz?Ìkx?ìºI?q=ý>£^>`ñ:¾óô¾éõ5¿¯_¿Àw¿~ô¿ÙÜz¿Nk¿T¿o|7¿bÄ¿¼í¾ Aª¾ØQ¾ÍË©½OL=»>
q>6¦>¥=Ï>ô>%s
?¦?º÷%?Z_1?Fi;?;:D?{ôK?·R?! X?1È]?Gb?^1f?ºi?pl?ê#o?Ù`q?`Rs?Du?yv?/¾w?/Øx?·Ìy?¸ z?X{?È÷{?Ñ|?où|?a}?ßº}?ª~?L~?n~?ÿ¸~?Ëä~?»
?+?H?±`?	v???f¦?h²?Î¼?ÏÅ?Í?]Ô?6Ú?Fß?©ã?vç?Àê?í?ð?3ò?ô?§õ?
÷?=ø?Hù?.ú?öú?£û?8ü?ºü?V#{¾ùTñ¾Þ¾C°>{?©¼>ày¿Ç>>&?88a¿Ë¹y?{¿Z·n?ç¸9¿¾j>úÍ ?æ*~¿A8? Q?Æl¿è0¾Är?áYà>aF¿]X¿{½>Ýq?(I?½ ½mÇH¿y}¿ËG ¿ÕõÊ<ýs ?eu?¢v?Þp1?»>Q6¾¬#¿uW¿z¿¤¬}¿6f¿;¿8/¿të¾¦ÚW½I,>g¼»>	?	É-?^J?Ý9`?¢o?Ìy?n³~?\ø?¾~?y?Ï8s?ù[k?Úpb?ÎX?!¼N?¶rD?:?Üå/?à%?Æ#?î½?
¹	?|?¢Ññ>³Dâ>aÓ>C®Å>¸>àQ¬>$È >qö>Ô>X>'ór>_b>1äR>YpD>þó6>Á_*>ý¤>Çµ>ñ	> >¡Zî=ßÝ=õÎ=:À=üê²=D¦=iü=~>=>=Mßy=#h=jX=ÇfI=Àm;=l.=R"=t=½="Ñ=yó<Bâ<"ÙÒ<Z6Ä<¶<¼ê©<<%<ÿí<òØ~<º'm<ç°\<¦^M<¬?<Ø1<B%<}.x?.Ça?æùu?ep?3G>«üm¿§¦^¾µw?â¥Z¿ió>ÃQa¾Æ¿<>Eð¸¾à00?õ0y¿l<]?ô½ónQ¿TR?ÏÃ>t¿y}¢¾Kf?)Î!?¿}¿4Ù©¾ÝV?÷ß?ëÑ?D¾vG¿âë¿òyG¿|¾³>E8? Hv?èè{?fáR?Ìì
?PR>¶	¾Mòß¾5.¿ Z¿Þ;u¿ï¤¿ÅY|¿q+n¿ÕßW¿¦ù;¿ä¿W÷¾"´¾õd¾¥Î½5%z<Ïû==:c>½>FjÉ>ËÕî>¼ ?Yü?%$?oÄ/?ôÿ9?3ýB?ÏÞJ?ºÄQ?ÌW?9]?Ó¥a?¥e? i?E&l?¨Çn?¶q?Ís?àÅt?§Dv?¼w?Ä°x?ªy?z?Ñ>{?á{?n|?¹è|?R}?S®}?Ëý}?£B~?F~~?î±~?­Þ~?n?ÿ&?D?@]?s?ñ?L?v¤?º°?Z»?Ä?Ì?kÓ?eÙ?Þ?ã?îæ?Jê?3í?¸ï?çñ?Êó?mõ?Øö?ø?"ù?ú?Úú?û?$ü?¨ü?=ê.?
÷Ù>ög?#_j?±Y?H¡¾¬h¿!ìA?Q{¼8B¿QVB?M¿ô3?HsÐ¾7Ô¾J.C?¹{¿òJ >Å¬E?ÞGL¿ð¿h[?ç½?	$¿@m¿1²Q¼a?µ]?û³Ð=¸3¿Hë¿òÃ4¿Õè½9?Imn?Yùz?UÄ>?À¼­>é½#¿àN¿òvw¿¯¿k¿èÂB¿¸¨¿zº¤¾Y±½ÜH>«£­>³£?qð(?!¼F?r]?±»m?!ax?Þ3~?òÿ?~?ógz?íCt?Al?Îc?ÓAZ? <P?÷E?h£;?d1?¶U'?7?Ü?;?2Y?\-ô>ä>µ«Õ>­«Ç>e{º>L®>Èn¢>>ñF>³>æ{u>¬½d>$U>F>Êà8>Z+,>Q >1E>(ù
>Õ`>Äàð=X9à=u¶Ð=zDÂ=ýÐ´=ÃJ¨=®¡=ªÆ=¥«=ò|=Âk=Õ¶Z=ÓK=Õk==MG0=ò$=§©=l=G5=hö<'üä<:Õ<LÆ<À¸<k¹«<«Í<¸µ<àb<xÇ<­o<å	_<íO<%A<d¼3<öA'<¨ì:?Ë¤g?ÛfU?føÍ>v´¿÷r¿µÕ>'?Gø¿#øY?ù£&¿5?È6¿¥Òi?L´}¿Ï¦%?cîG>x!s¿Òª"?9J?íú[¿Âi¿ÒgJ?D?VÀ¾¢ú¿§ìñ¾/úÿ>Óª~?Cs6?=øÍ¼ÝE5¿­F¿æU¿ìlº¾éI>Q·*?Ïp?)V~?ÌB[?pÊ?`>^P°½ñÛÊ¾%&¿ã;U¿ncr¿Ð	¿}¿Ôp¿¡[¿W@¿Ü^!¿q ¿¬ï½¾>ýw¾ÏBò½êÂ§ºP{Ü=}åT>Ú<>÷Ã>ëé>LÊ?/é?ÂO"?$&.?8?\½A?ÆI?¦ÏP?öV?xT\?a?Ke?M¤h?»k?ujn?Á¿p?Ær?Ýt?µv?Ïbw?ðx?y?'ez?Ý${?Ë{?
[|?Õ×|?öC}?¥¡}?Ïò}?9~?v~?Ëª~?~Ø~? ?\"?@?ÅY?p?U?	?¢?¯?â¹?HÃ?kË?wÒ?Ø?ÚÝ?nâ?dæ?Óé?Ìì?_ï?ñ?ó?3õ?¦ö?ç÷?ýø?íù?¾ú?rû?ü?ü?hÌ{?ußz?+~?$l?õú>éY¿ñ¾~?°Ì¿d¡ë¼ù»>ýì¾<T³>öAÜ¼áõ¾å;o?u_¿×Å<ÀLj?MÊ¿ó2¿Ì8?´¸B?q÷¾2z¿5¾VeL?ú¶m?RÝn>þ¿Ëô~¿:*G¿¸2¾ñ>Àxe?9~?MK?nåÎ>I½¿Oí¾2F¿q²s¿ìÑ¿go¿¦I¿bê¿EQ¶¾oö½mIØ=qd>(û>Äû#?HÚB?ÄZ?¥»k??&w?º}?õ?è~?${?ëCu?5Ém?b#e?¦®[?¶Q?
xG?ª#=?fß2?nÈ(?±ô?0t? R?Ú?_ö>»æ>éÆ×>/¨É>Z¼> Ö¯>ñ¤>J>¹>Å><x>pg>ÓOW>H>jÍ:>Ðö->äý!>Ô>Mm>»>Ðfó=â=ææÒ=`NÄ=ó¶¶=:ª=ìF=ÒN=³=.=Z|m=¤]=Ü®M=çi?="2=ÄÅ%=ØD==k=H§ø<eç<RU×<ßbÈ<ïyº<­<>|¡<hF<Á×<x"<`3r<ãba<3½Q<-C<¶ 5<©)<ÂÂ8¾íéK>9kô=ÓÃ¾ñv¿»Z¿»a?_½Ú=)uV¿âä?5n¿}-c?¨Èo¿Oè?vÂ`¿'>¶>+û>ãì¿LÎ>sÍH?X7¿Á(1¿÷0&?7`?>¼R¾Añ{¿,#¿C¾>ðx?EFK? À¸=åÖ ¿|¿Ø¸a¿øâ¾²äð=·ñ?ë*j?ó°?H×b?
."?´Ð>)½ôZµ¾JÉ¿!O¿A8o¿M#~¿~¿XGs¿G_¿D¿	&¿xL¿È¨Ç¾×v¾.1¾;
¼6½=F>©µ>ñ«½>Pä>èo?5Ò?[v ?,?$7?»z@?æ«H?RØO?ñV?î[?·^`?+d?À'h?¸Nk?Qn?úmp?~r?9Kt?9Úu?h4w?²`x?ey?æFz?£
{?M´{?WG|?ÃÆ|?,5}?Õ}?¶ç}?/~?´m~?£~?>Ò~?©ú~?¬?þ;?@V?ÿl?±?Á? ?R­?f¸?ÿÁ?NÊ?Ñ?¼×?!Ý?Îá?Úå?[é?dì?ï?Kñ?Dó?ùô?sö?»÷?×ø?Ìù?¡ú?Zû?ùû?ü?ò[Â>¿>?gó;?3£º>H×í¾.M}¿gI!>Rþ`?h¿Só>ÍÐä½[qÔ¼]°½Ì·>f¼B¿÷ì?Èö,¿& ¾×v}?xÎ¾ðòX¿,?È`?ÚÍ¾ã¿s¬¾¶U2?æØx?m¸>A¿çz¿LDW¿5$¾úÆ>÷ËZ?cÒ?#"V?Jï>C<¬zÑ¾¥<¿#Co¿äû¿µCs¿£P¿õð¿Õ©Ç¾2À¾ÃÉ=â>Øî>Ôë?bÜ>?+W?'¢i?BÔu?÷|?HÙ?y=?Ô{?À8v?Êñn?ðpf?ú]?¶+S?ôH?K >?¸W4?¸8*?0Z ?èÌ?´?rÑ?©ßø>)ôè>ûàÙ>Æ£Ë>ö7¾>[±>º¥> >È+>>h>(z>àxi>>Y>Y J>Þ¹<>!Â/>,ª#>Àc>^á>C>Ãìõ=Éìä=FÕ=9XÆ=ß¸=¨Ó«=#ì=óÖ=¼=ë=íôo=nP_=áÒO=ögA=¿ü3='=à=Ç=ý=&>û<îÍé<hÙ<!yÊ<k¼<ÈV¯<Ð*£<×<¢L<w}<2¹t<á»c<yìS<6E<7<\Ç*<xÖl¿¹¼*¿ÊÏ-¿«bn¿È³b¿J>úÍ|?Ó>ô¾÷ÊÕ¾4Ta?®e~¿öé?nz¿[o?£,&¿ÑlÅ</»<?·v¿uÅ>Xj?·æ¿+PU¿üõ>s?Å6ñ¼³q¿%ª7¿"^p>³Én?óø\?DQ>ç
¿]v¿Á÷k¿ìç¿ÓË=¹K?ybb?Î÷?Åi?b-?¶>n7<¦z¾¿%¿xI¿l»k¿ªñ|¿ÁK¿­u¿pb¿ì²H¿m*¿¾
¿WLÑ¾â¾°5¾Ç½¡­=,8>Ê'>rÁ·>mß>¢?x·?e?ß*?±5?T5??¯G?¿ÞN?ÞCU?ÙZ?Ì¸_?¨÷c?û©g?Váj?=­m?`p?Ä7r?õt?0¤u?w?
8x?ÚAy?V(z?%ðz?V{?q3|?µ|?;&}?ã}?Ü}?Ë%~?Ie~?I~?íË~?1õ~?ï?ã7?³R?ëi?~?r??«?ç¶?²À?.É?Ð?äÖ?fÜ?,á?Nå?âè?ûë?ªî?ýð? ó?¾ô?@ö?÷?°ø?«ù?ú?Aû?ãû?qü?(É¿öÏ·½ñ)½¸³Ö¾jq¿¥D*¿äû8?Ýç>É}¿8ÙY?cR¿ø$Ô>õòî¾û/?û<q¿ÈXs? dÓ¾`ï¿p½}?"¾¹Ír¿ Õ¶>át?c­ù½å/}¿èù¾?Ýà~?6mö>ÒÉ¾wær¿âd¿'£»¾Õ>i{N?§Â?Ú
`?¢?µ4¥=Ôá´¾62¿#,j¿~¿ v¿ë*V¿E¹%¿=¾Ø¾	@¾F.=->D[â>yÁ?Ã:?²vT?pog?Jkt?ç9|?«?l?Mv|?_"w?óp? ¶g?Åt^?zT?­kJ?C@?Í5?¦+?¯½!?þ#?öå?ö?36û>Ö+ë>èùÛ>oÍ>À>úW³>Í_§>$>¹>tÂ>ª}>ùÕk>cº[>ê¯L>$¦>>N1>VV%>äò>]U>âp>rø=dFç=G×=bÈ=Áº=­=T¡=_=Àò=Û>=zmr=3a=âöQ=fC=t×5=c9)=6{=s=°a=Õý<Ï6ì<~ÑÛ<aÌ<L\¾<v%±<bÙ¤<Èg<Á<uØ<?w<ßf<¾V<y>G<Zi9<,<à¼Q¿÷~¿Ç¿gh¿:®©¾µ*??G÷0? Ud¿Y>^t?ØpS¿¢ýh?(ib¿ië9?`Y«¾Ôù¾v)i?DLX¿ìÃ¾ÖÀ|?C¢¾o¿A>(~?ÿa>X_¿WØP¿¿=æe`?¢Ek?;¯¡>nSå¾k0n¿î+t¿ÎT¿/1½¸·÷>ìY?m*?ó}o?«b7?ëßÎ>,u=F¾Q>¿C¿ îg¿Cu{¿§Ä¿w¿we¿ß®L¿³/¿ÎÏ¿MÙÚ¾ù@¾x-/¾P½ôn|=§)>>´Ï±>,«Ù>_ý>?ì¸?T7)?þ;4?-í=?ünF?ðâM?ÌgT?Z?P_?Ãec?þ*g?ârj?:Mm?ôÇo?Rïq?Îs?mu?/Öv?øx?>y?u	z?aÕz?#{?V|?¤|?#}?Ïz}?*Ñ}?ú~?É\~?ì~?Å~?«ï~?&?¾3?O?Ïf?V{???×©?cµ?c¿?È?Ï?
Ö?©Û?à?Àä?gè?ë?Nî?­ð?»ò?ô?ö?b÷?ø?ù?gú?'û?Îû?^ü?Ì¿3Z¿Ø¶I¿1 r¿uj¿Ø+ô¼W~?fç	¾yÌD¿*ã?}b¿_E?®I¿¶´i??í¿¤ïJ?$ß½ÚÈA¿Hk?tÃ=x¿Eo>ô~?¤ó=òr¿@ë¿÷ä>7¯?jô?^¾»÷g¿èÜo¿%Bê¾W>·@?ù	~?0±h?ÞÚ?©>ð¾P(¿qd¿ý}~¿¬{y¿·ã[¿C@-¿©é¾!b¾k33<¿g>x²Õ>}?Á6?¾BQ?½#e?yër?Vh{?áj?Ý¹?A}?¾ x?¥&q?gôh?ýÍ_?ÌV?ÈÞK?A?W?7?í-?*#?py?â-?dE?ûý>bí>­Þ>)Ï>ñÁ>üµ>}©>¯>X>h>À>¼2n>Aï]>B¿N>=@>WX3>c'>ð>IÉ>sË>^øú=ëé=×wÙ=ÄkÊ=h¼=k\¯=~6£=&ç=À_= =æt=ôéc=àT=dE=&²7=0ó*=b==ÑÅ	=ï5 =¯î<Þ<¡¥Î<zMÀ<#ô²<ó¦<wø<b6<t3<ÕÄy<Ümh<KX<ëFI<¬M;<ÁL.<8#=tâ¿h¡¿"þ¦¾ÅÍ>ââ?¢²è=üª}¿ç¹#?äýò<f§î¾	#?¬¿bùÐ>õÃ<æ¿_x~?fH'¿Ê¾Ö~?,®½*U}¿ÿ¸=d?G¤>¡çG¿3e¿EPK½N?ôu?Ø>Äç²¾¢c¿3Cz¿[(¿Uý½sÕ>VO?¯I}?Ot?"A?¸ç>\ôÝ=ve¾¿¼w<¿©Ñc¿®y¿ü¿[y¿¿¢h¿ÌP¿Åw3¿w¿¤Nä¾ ¡¾A¾>½Ss==$*>jò}>ôÖ«>|MÔ>hø>ðv?ùÔ?Ú'?sÃ2?J¢<?ÏLE?èäL?¾S?«WY?Dh^?|Òb?Êªf?\j?Gìl?·so?)¦q?s?~6u?\¦v?}åw?Eúx?Eêy?Wºz?´n{?|?|?ã}?m}?¸Å}?~?2T~?{~?¿~?ê~?P?/?{K?«c?x?Â?w?¨?Û³?¾?æÆ?Î?-Õ?ëÚ?ãß?1ä?ëç?%ë?ñí?\ð?uò?Fô?Øõ?5÷?bø?hù?Jú?û?¸û?Kü?.¡¿·m¿Ãz¿iÒc¿Ë¾'?d)\?,¿Wf¾$pa?Üÿ¿Ôx?«x¿Kæ?OÝl¿ºP?¦NR>Jùi¿ÙóF?(¬¯>r}¿ùã½ãO~?¿>*`¿ª>¿>º?{?|ÿ1?í ¾üñY¿Ïx¿øD¿Aó=yP1?3«z?³p?^ö#?HV>± s¾lé¿Ç^¿×|¿¶Ó{¿m8a¿÷4¿^ú¾0ó¾p©¼vLJ>¦àÈ>ö ?8@2?´ôM?N¿b?òTq?kz?Ð?Çà?ç}?ÒÓx?Õ2r?;*j? a?¤jW?^MM? C?®8?Îz.?~$?:Í?ut?º}?ûÝÿ>äï>G(à>ðÑ>)ÍÃ>`×¶>®¨ª>>9>£>v>µ>'p>Ù#`>aÎP>(~B>:#5>R®(>ä>!=>ó%>~ý=]ùë=¨Û=vuÌ=dN¾=¿ ±=¡Û¤=7o=¼Ì=`æ=^w=°6f=Ù>V=bG=Ö9=û¬,=± =Æ=ð)=\=ñ<¥Mà<ß»Ð<¦>Â<ÐÂ´<6¨<&<B«<r<¦J|<ÙÆj<IzZ<^OK<þ1=<t0<îÀ\?*¾>N>¼é>ªçj?ùH?ç¡¿º'=¿Òps?ò¾.l;©­p>f?u¾Ænå<:Â>ÄÅV¿óz?ÅÏ¾$!¿¥up?ÆÖ>ÞA¿2Ëê½Æew?ÔF÷>µÔ*¿æs¿¬]D¾·ý7?Ú|?5L?q×|¾dÍV¿0~¿f¦8¿=äO¾9ü±>D?Wz?(¦x?FJ?«þ>Üe >8#8¾>hó¾5¿ng_¿ýw¿ûñ¿`ôz¿_yk¿@T¿¿7¿ñ¿U«í¾-Òª¾
ñR¾Fß«½bØü<¥>1²p>n×¥>èÎ>cÀó>@Q
?í?)Ý%?ðG1?°T;?,(D?ªäK?¶©R?X?¨½]?Õ=b?`)f?Æi?dl?©o?I\q?jNs?Ôþt?vv?»w?ðÕx?ÅÊy?	z?	W{?ö{?¸|?|ø|?@`}?(º}?~?K~?÷~?¸~?rä~?n
?S+?ÑG?`?Þu?`?h?J¦?O²?¹¼?½Å?Í?OÔ?*Ú?<ß? ã?nç?¹ê?í?ð?/ò?	ô?£õ?÷?;ø?Eù?,ú?ôú?¡û?7ü?íoç> ¾æsö¾¾ªE«>w¤z?¶¶À>òly¿*{>}¦?Vþa¿z?üÞ{¿0o?·:¿/o>ÜÄÿ>(~¿ßø?h?zál¿¾Å¾íûr?ô1ß>+ÂF¿53X¿É>­©q?qáH?»½
I¿l}¿¡ ¿øÔ<® ?ï«u?Òv?E1?þO>Ò7¾K¿äW¿z¿b§}¿&f¿~;¿^¿g³¾­'V½ü¬,>Dè»>¬	?Ø-?üJ?dB`?Ü§o?8y?Ü´~?(ø?9~?y?z5s?Xk?lb?ùÉX?g·N?îmD?È:?+á/?Ü%?X?«¹?õ´	??>Êñ>´=â>ÄÓ>¨Å>%¸>^L¬>ùÂ >ñ>Ï>ÓS>9ër>)Xb>FÝR>äiD>÷í6>#Z*>À>å°>d	>Ê >ºRî=&ØÝ=Î=%4À=å²=¼¦=C÷=³9=:=þÖy=hh=ÏbX=`I=g;=Ãf.=·L"=n	==ÈÌ=lqó<·â<ÒÒ<Ò/Ä<|¶<å©<Õ<! <pé<vÐ~<Õm<©\<ÐWM<O?<&Ò1<nZd?Few?d`?'6u?{@q?õeP>w,m¿ñf¾.x?@ºY¿ð>^[¾Ñ+7>y¶¾JL/?Míx¿çÄ]?´uü½çP¿jR?A&Â>UÍt¿	.¡¾1ef?kW!?í	¿Mñ|¿_ð¨¾±?Ü?è?ô¾QÓG¿Ùé¿0KG¿òö¾Í>ä°8?\Wv?ß{?ðÅR?óÆ
?/pQ>1Q
¾3à¾ÛM.¿ñ°Z¿BDu¿]¦¿U|¿Ë"n¿	ÔW¿Êë;¿Ý¿_îö¾B´¾ï¹d¾AÍ½u}<ù0ü=gc>^Ñ>|É>(æî>(?Ñ?H+$?zÉ/?c:?C?8âJ?µÇQ?°ÎW?]?Î§a?À¦e? !i?'l?ÊÈn?²q?¨s?Æt?KEv?Kw?@±x?õªy?uz?!?{?Ìá{?Än|?íè|?ÆR}?{®}?îý}?ÁB~?`~~?²~?ÀÞ~??'?D?J]?s?ù?S?|¤?À°?_»?Ä?Ì?nÓ?gÙ?Þ?ã?ïæ?Kê?4í?¹ï?èñ?Ëó?nõ?Ùö?ø?#ù?ú?Úú?û?$ü?ø¬~?#?+¶>Ñóû>Ôc?Xa?¿X¾en¿q9?VZú<^r¿ÉÎH?S¿Û:?/Sß¾Õ½>?§G|¿áí«>ýA?WrO¿ðNý¾r]?|S?'¿7Ôk¿¡Ô::_c?D2\?öº=j5¿ÕØ¿ý*3¿Ã:½[×?|o?î¢z?Ó·=?ýª>£ßó½8¿;O¿¾w¿¼õ~¿¬j¿)0B¿ ù¿@H£¾$°«½:ç>ÐË®>K!?ÍV)?G?F­]?_äm?Ñyx??~?  ?0z~?÷Wz?.t?Ü}l?É±c?À#Z?ÛP?û×E?ß;?E1?U7'?Ço?ý?ë
?K??!üó>ïQä>¡Õ>Ç>HTº>ï­>JL¢>;b>¬(>¹>ñFu>2d>ñëT>qUF>¸8>Õ,>. >$>ÄÚ
>D>¬ð=4à=²Ð=ÜÂ=N©´=Ñ%¨=H=¥¦=×=tO|=Ðj=ÁZ=^K=/B== 0=Þç#==-ò=3=HÚõ<ÈÉä<ZèÔ<þ Æ<'`¸<¤«<ª< <nD<#«<Òxo<ÓØ^<B`O< ú@<Ø3<*Ð=dH??o?ÓÛ^?NXì>CCò¾O*w¿=¼>ù{0?dá¿	T?È¿Íß?8Æ/¿$]f?~¿¸ +?ï->"q¿Ó'??^y^¿°q ¿PM?»ôA?4Ç¾üÿ¿O;ì¾ñ?­ï~?4?´½Ú6¿¶l¿qT¿·¾}P>á+?FMq?³-~?4Z?ÜÖ?²ý>îl¸½àÌ¾oÐ&¿Î¯U¿¢r¿N¿²~}¿p¿#D[¿ý?¿Eü ¿` ¿#½¾Ppv¾ÓNï½èv8ª
ß=zV>Å>­	Ä>Þê>Pû?²?Av"?H.?i±8?×A?ÝI?¿ãP?W?Éc\?ha?ë"e?j®h?ÔÃk?rn?eÆp?FÌr?Þt?v?fw?4x?Öy?gz?þ&{?àÌ{?£\|?7Ù|?)E}?°¢}?¶ó}?ç9~?µv~?a«~? Ù~? ?½"?b@?Z?Jp??9?ª¢?,¯?º?bÃ?Ë?Ò?£Ø?éÝ?{â?pæ?Ýé?Ôì?fï? ñ?ó?8õ?ªö?ê÷? ù?ðù?Àú?tû?ü?/|?Õá?Aq?.x?;¸v?ó½>ÖJ¿çH¿ûz?Ì1ò¾gÜ½õÞ>@F¿kÑ>E°½ÊñÛ¾>j?­Òd¿h=DÕe?Èf&¿G,¿ø>?üx=?Ï#¿¿y¿CX¾íO?'k?ÁZ>e¿ R¿D¿ÕA#¾;Ý÷>Ìïf?eÖ}?¯=I?~òÉ>n?q½qñ¾rG¿.Ot¿¾¿¯Æn¿BH¿Ã®¿Õ¬³¾~ ì½â=Ï¡> ý>4½$?>rC?9 [?¤
l?NWw?m·}?Mø?ÆÙ~?õ{?õu?m?Oðd?òw[?±}Q?<>G?Øé<?I¦2?(?¾?ø?? ?f?,ö>ødæ>u×>k[É>È¼>4¯>1Õ£>Ò>>gÙ>O¢w>ñ¿f>aúV>Ï@H> :>h±->-½!>1>5>2>7ó=28â=;Ò=ÿÃ=m¶=ßÊ©=H==á=ãÇ~=Èm=¯ª\=\M=Ø?=NÚ1=%=¹=JV=c="Cø<Øç<þÖ<(È<Ò.º<3B­<1;¡<ß	<l<î<ÎÑq<a<´hQ<ñÞB<W5<I@H¿Îø¼pB«>JÌ}>¾ »m¿-¿"ÑT?½I>	a¿b~?+¬f¿Z?´i¿Í?í,g¿sÎ>ôå>Îe¿sá>¹B?2[=¿æ*¿È&,?t\?m¾¡ÿ|¿?W¿«FÈ>z?MRH?#=3$¿¼¹|¿z `¿òÜ¾WÑ>ø;?É>k?:?¿a?¿} ?õ>­7½
¡¸¾B¿ºeP¿¸o¿óJ~¿²o~¿ìr¿Ê^¿óC¿V%¿¾¿2Æ¾_	¾Ow¾Á{¼¢ØÁ=3²H>²>¾>©å>+Ë?F#?¾ ?ÐÃ,?Ç[7?¦«@?ÆÖH?ÕýO?·>V?´[?¥w`?ád?¥:h?'_k?n?bzp?Er?Tt?Vâu?s;w?Ífx?gjy?}Kz?{?À·{?TJ|?ZÉ|?k7}?Ç}?eé}?÷0~?÷n~?¬¤~?1Ó~?|û~?b?<?ÉV?um???Ó ?­? ¸?1Â?yÊ?¦Ñ?Ü×?=Ý?æá?ïå?mé?tì?ï?Wñ?Nó?õ?{ö?Â÷?Ýø?Ñù?¦ú?]û?üû?±«¤¾ÿ ?0]?ßY?ü?Ìö¾ý¿»Hu<O9o?>Y¿½>ë» ¾î<, >½4¿à¾~?±9¿¯P¾MÕz?×è¾ Q¿{¿?¸Z?¬.²¾e¿Ü¾ªw8?Óv?urª>ò@¿Ù{¿oìS¿O>¾70Ð>hK]??ÆÈS?è><Ä:¯×¾+Ç>¿Lp¿þÿ¿tr¿L«N¿&1¿NÜÃ¾4*¾¡¦=Ê0>ñ>ä ?À??;X?Ùj?Ä v?}?á?÷+?®{?¹v?2±n?(f?Æ\?ãÙR?ª H?­L>?þ4?¸ç)? ??ìS?ò?q[ø>Ëvè>njÙ>í3Ë>¤Î½>Y4±>­]¥>|B>*Ú>Ý>Rýy>fóh>Y>ü+J>KM<>Ü\/>¾K#>¹>U>ÒÉ>U^õ=hä=·Ô=(åÅ=º1¸=åo«=B=|=>5=& =qio=Î^=
ZO=~÷@=3=('=]=eº=¯=û«ú<çEé<ÑÙ<RÊ<|ý»<Âð®<ÞË¢<½~<iú<ó0<É*t<[7c<&qS<BÃD<<7<ñer¿õØQ¿ÅI ¿û¿ý"Z¿S/s¿Ì¼¨ø?ºK¶¾Ø¿ØÑm?Pÿ¿ù}?2ä¿Ûv?ÂL5¿!~Ê=®80?« z¿o±L>Q0d?¿-N¿É?Xÿo?A½mt¿1¿Ù>Ýeq?oZY?­7>ª¿èÖw¿ái¿9´ ¿Pa=;Ñ?n2d?íÿ?--h?&´*?z°> ð9+U¤¾b¿ÔJ¿Bl¿;}¿L(¿é
u¿q¶a¿ÍG¿V)¿Ì		¿!.Ï¾åÏ¾£=¾­û¼j¤=ÉI;>U>	¹>®-à>§?.?à?©<+?6?R}??ËÍG?øO? tU?»[?Ý_?¤d?ÒÅg?ùj?IÂm?©-p?¥Gr?¾t?&°u?éw?
Ax?¨Iy?/z?öz?l¢{?Ù7|?V¹|?)}?Á}?úÞ}?ñ'~?&g~?ç~?SÍ~?gö~?û?Ì8?|S?j?~?ô?ø?ù«?<·?üÀ?nÉ?¾Ð?×?Ü?Pá?må?üè?ì?¾î?ñ?ó?Ëô?Kö?÷?¹ø?³ù?ú?Fû?èû? ux¿öd¡¾@B7>ÂU>wJ¾Y¿­5M¿©?ÔÉ?ß¿2C?fCâ¾¦D>_ðµ¾½¼?SÕf¿Qüy?·2ý¾pì¾õ¡?SÁl¾ø§l¿	Õ>·p?î4¾Í~¿¸áã¾:?ª}?åä><Ú¾fxu¿¾3a¿¿­¾Éó¦>T7R?ó?ôK]??s}}= ?½¾5¿ü¶k¿[»¿3´u¿ßoT¿8~#¿áÑÓ¾¸6¾¹ÉS=S·>ßþå>C?ö;?_U?+h?NÖt?Úr|?Kº?½p?¢H|?Ëßv?©¾o?Yg?s^?h1T?>ÿI?X¬??a5?¾<+?ÒV!?°À?¢?â°?×ú>dê>Z^Û>¡Í>Ú¿>øÕ²>¼å¦>²>2>^>ùW|>&k>[>úL>o>>01>5Ú$>->é>d>^·÷=ùæ=$¥Ö=½ÊÇ=âõ¹=ä­=6¡=`í=ì=XÜ=¶q=ò`= XQ=!ÒB=ÑM5=K¹(=ÿ==nú=Óý<ôë<+Û<{ôË<&Ì½<P°<\¤<ó<fU<Ús<Äv<fe<yU<§F<îÜ8<ø½v¾Ðòr¿÷Ý{¿fz¿§ñz¿Éñ¿a?4vP?¨L¿^Û ½Ö%?Y¥e¿
át?²Jo¿m²L?`Ý¾&©\¾f^?&.c¿Z[½Iy? >Ã¾Éh¿ã!²>æø{?EúÅ=Ö>e¿dJ¿ä
>6þd?d}g?êT>yó¾Ïp¿ùr¿¿n£¼H³ ?Ï/\?a?çÞm?ér4?Ê¾Ç>l;=¾¾õÙ¿þD¿i¿:ë{¿V¨¿Ãúv¿·d¿ K¿ìÌ-¿s¿ÏØ¾¾9ù)¾Þ<½W=ØØ->|>¥³>8Û>¢Áþ>®6?D?ª²)?¨4?L>?§ÂF?-,N?Ï§T?fQZ?	B_?4c?ðOg?j?)im?:ào?fr?^às?~}u?öãv?ëx?(y?qz?*Ýz?ã{?/%|?*©|?}?~}?vÔ}?Õ~?B_~?~?hÇ~?Fñ~??ó4?'P?·g?|?Ê??Yª?Ôµ?Ä¿?`È?ÕÏ?IÖ?àÛ?¸à?éä?è?¯ë?iî?Äð?Ïò?ô?ö?o÷?ø?ù?pú?/û?Ôû?&:¿º
s¿.+"¿jÂ¿³P¿ªf}¿:È¾²¸p?:é§=á§a¿×ôy?÷@J¿·S(?Ï/¿'³X?~¿Go\?{b¾n/¿âës?«Yz;I|¿
?c>ö|?²ÒW9c>w¿\Á¿Þý>äÿ?8é?Ñ£¾¹Bl¿;l¿ÀÙ¾îx>úÅE?"Ù~?^»e?x?öù=H5¢¾æ+¿åf¿Mð~¿ñx¿¶àY¿{*¿Øã¾âU¾|®¶<ÿGr>ñCÚ>Öd?8?nlR?Ìùe?xs?þµ{? ?¨?A×|? ²w?ñÃp?h?°R_?9U?ñYK?ÓA?º6?¢,?^ "?íþ?2¸?âÔ?¹´ü>Âì>FQÝ>âÎ>iFÁ>w´>^m¨>^!>> >C²~>rYm>K$]>ÆN>ká?>e³2>h&>ò>£C>êN>Rú=ÂÇè=®Ø=G°É=º»=Ü¹®=%¢=@Z=Ü==³t=ac=óUS=Â¬D=7=lT*= ==ÖE	=©}ÿ<Âí<DAÝ<£åÍ<Ï¿<ÞM²<8í¥<zh<c°<Á¶<¿Üx<ãg<	W<äH< :<½/?«Ô ¾F¿ÖÒS¿ ¿ø>ýu?S;®>]#¿õÉñ>q0]>ð¿Õà@?>P:¿UL?üWÛ½×+¿ô¨y?Í:¿ßi¾ÿ?ÊÏ-¾Ùy¿ÿo'>  ?Ã>PQ¿pi^¿$ñ:ûU?ör?&Å>KÅ¾»±g¿Rx¿,"¿µÂ½ééá>?S?
~?Ðr?4³=?ÝtÞ>5Kº=ÊÌu¾:h¿aä>¿çTe¿xZz¿´ï¿4»x¿°g¿s*O¿×ç1¿Í¿Iëà¾:¾å¨:¾ÆÙ{½T=ý_ >UY> ý­>û;Ö>qMú>;?H?Ø%(?K3?=?_µE?s@M?ÅÙS?Y?2¥^?c? Ùf?+j?9m?o?Àq?s¥s?]Ju?·v?rôw?=y?õy?Äz?'w{?Y|?×|?b}?\r}?ØÉ}?£~?JW~?*~?mÁ~?ì~??1?ÊL?Íd?y??4?¶¨?h´?¾?PÇ?éÎ?}Õ?/Û?à?dä?è?Lë?î?yð?ò?\ô?ëõ?E÷?pø?tù?Uú?û?Àû?m7=>c­Q¿kô¿Ò]|¿Kç}¿É*¿¸>w?µì¾¦
¿ØÆv?jz¿h?xi¿­{?ry¿û)?%Î=5
[¿¶tX?\t>½­¿üù<cè?TX5>dîh¿¥2¿þf»>Ê}?11'?§ªU¾GS`¿æt¿ï¿">8?P|?m?Î?æ!:>I¨¾<¾!¿Ûâ`¿@}¿âz¿«û^¿n1¿üò¾Äpu¾ä{é»ÖòV>»dÎ>|p?(4?¥bO?ìÈc?r?~çz?1>?þÑ?_Z}?²zx?Áq?[¦i?6`?NÒV?¼°L?bB?8?_à-?*è#?»;?è?ï÷?ßþ>â¤î>0Cß>¸Ð>NÃ>¶>ô©>J>Sâ>æá>>o>Ê1_>aìO>?«A>z^4>×ö'>Öe>±>a>/iü=z÷ê=Õ·Ú=ÅË=~½=Ì^°='¤=Ç=<0=²T=MOv=?:e=âSU=`F=LÁ8=ï+=? =±æ==
=?ó = ð<}Wß<ËÖÏ<wiÁ<lü³<ä}§<WÝ<_<©ù<º5{<&Åi<zY<4pJ<Rb<<q{?Gß?¼ê+¾~È>g´>?}Én?,¾0
c¿/|Y?Ñ1¾¸2R¾Ê*Ö>bÒ¾þ²G>Sôk>°D@¿´Y?=¿i¯¿©x?Ç)M=ßô¿iâÛ¼ ô{?ÞgÔ>,ë7¿B<n¿äG¾àÜA?t]z?þ°ö>Þ#¾&\¿ÁÅ|¿²ð1¿Ê -¾¼SÁ>]kI?>¼{?Ðüv?nF?ë£ô>?>µ¬K¾û¾8¿~Ua¿ x¿Vþ¿åKz¿NGj¿t¬R¿ì5¿O¿¼ªé¾AÜ¦¾{KK¾c½÷s=Öß> bv>ªj¨>9Ñ>ØÒõ>d=?þ¾?;&?ë1?ä;?ó¥D?ÏRL?
S?%èX?^?¾}b?af?.Ãi?z´l?;Co?|q?þis?Ãu?Õv?Íw?åx?QØy?Çªz?6a{?Uÿ{?]|?ÿ|?þe}?!¿}?Z~?@O~?2~?d»~?ßæ~??%-?eI?Üa?w?f?K?§?ú²?L½?=Æ?úÍ?¯Ô?}Ú?ß?Þã?¤ç?çê?»í?.ð?Mò?#ô?ºõ?÷?Lø?Tù?9ú?ÿú?«û?úCm?¢ï¼")¿'ÅM¿2#¿:.½)åY?Â&?®]¿½s9:?TDx¿Ïü?â¿x[|?4W¿?³É>·>9¶v¿/?¶ë>û«v¿K>¾¾ñy?@a²>0AT¿
-L¿=ßj> w?çæ=?
µÁ½OÍQ¿£{¿2Þ¿=C )?å\x?@6s? ¾*?Ðv>\U¾{$¿RªZ¿ãÈ{¿þÎ|¿»¾c¿þ8¿7¿`¾¿¯½jt;>6cÂ>3g?H0?yBL?Âa?sp?gz?äè~?sî?óÑ}?y9y?Ðµr?¯Âj?ýÇa? X?N?$¸C?¹e9?ñ./?0.%?w?Ð?		?ï ?Á±ð>4á>ßÒ>»Ä>¢··>U{«>Üþ>Ñ9>t#>Þ²>N¾q>?a>ËÖQ>ìtC>n	6>)>Ù>¯÷>ËÓ>÷Áþ= 'í=ÁÜ=8{Í=#B¿=µ²=ï®¥=ð3=Þ=Ú=âx=^g=ÎQW=ûaH={:=©-=Þ!=ÈJ=¢Ü=©'=>ò<´má<òÇÑ<8Ã<ùªµ<©<4R<[f<<<´}<iôk<ê[<TL<%><ËBÀ>ä?	Ù??DL?ðùD?YÝ?a?<B¿p¿¯Ý?ü¨/¿Éy>À¹!<-È÷¼H!,¾¦
?YLk¿fo?ª¾àÕ:¿§Kc?îó>m{¿oh]¾óõo?I?¸m¿Ï,y¿Ö¾­+?2Ú~?©±?kþF¾ïO¿}T¿-@¿Ax¾§Ì>Ë½>?/wx?az?ËN?>?i9>ê/!¾wÒé¾±ï1¿x]¿)yv¿9Ô¿¬{¿ûÔl¿ V¿¥Ú9¿\Q¿XTò¾õp¯¾Ñß[¾ÌÏ¼½½<þX>àj>uÒ¢>ó0Ì>ôQñ><	?À÷?Ø%?d0?F¬:?iC?AcK?8R?=1X?wg]?¹òa?ûçe?âYi?íXl?¬ón?ò6q? .s?±ât?§]v?l¦w?Ãx?Ùºy?;z?K{?%ì{?¼w|?²ð|?Y}?P´}?û~?!G~?*~?Mµ~?á~?ö?0)?÷E?ä^?zt?,?]?c¥?±?¼?'Å?
Í?ÞÓ?ÈÙ?èÞ?Wã?/ç?ê?dí?âï?ò?êó?õ?ðö?'ø?4ù?ú?çú?û?Q?ÀH?A>Ì¾i=?nÍ~?·íÖ=ËÙ¿bñ>Á¤>ÞC¿¥j?Àdo¿Y\?Õ®¿gµ=Ûê?ÿ¿~ô>â'?³a¿¡»Ã¾áj?
;?fÇ9¿cra¿Á?´=øk?ð°Q?ÿ©<¶Û@¿éÔ~¿_(¿30i¼?ßs?ì0x?/Ã6?¡>ú»¾Í ¿íS¿0ny¿YH~¿(h¿£l>¿ÿ¿Hä¾j½Ò>`A¶>³I?uè+?AI?'_?eçn?Èy?~?uý?ú=~?lîy?V¢s?Øk?ûùb?'`Y?RO?î
E?G·:?T{0?nr&?±?ÚE?-;
??[½ò>ö#ã>LbÔ>uÆ>W¹>©­>m¡>>Çd>uß>Iðs>Lc>ÁS>p>E>B´7>+>+L>Q>'
>T >´Vï=LÊÞ=`Ï='Á=¨³=Ë6§=Á =|×=ÿÌ=qèz=ïi=·OY=<J=¾4<=Ä%/=zþ"=Þ®=(=\=|ô<êã<¹Ó<ÆÅ<Y·<<ª<Ç<WÁ<v<®ç<¬#n<[]<Ô8N<´ç?<j¶¿Ú?Li}?é}?X?ÖfI?ÜÅ½~¿×=®Ãa?Nbr¿+Ö$?ÝñÌ¾gµ>P¿}sJ?gþ~¿ûDJ?ÂÌn; ë`¿ÐCA?@ñ>l¿¢Ë¾Mf\?!0?@ò¾®¿wÆ¾5Ú?ãñ?Y(?4pÃ½ó¸@¿]ù¿#+M¿þ ¾ßüz>aB3?ëIt?Æû|?<>V?»?s~f>Óì½¼Å×¾+¿X¿)t¿gq¿àÜ|¿H;o¿ýTY¿±=¿Áz¿Lçú¾u÷·¾½dl¾1Ü½Zb<*ï=e¤]>°4>O"Ç>âÊì>Á7?-?·n#?6#/?)r9?ÂB?ÍqJ?deQ?ÐxW?Æ\?fa?æme?©ïh?ük?i£n?:ñp?wñr?&®t?0v?à~w?G¡x?y?swz?¹4{?ÆØ{?óf|?'â|?èL}?e©}?ù}?ð>~?{~?(¯~?EÜ~?Y?1%?B?å[?âq?í?k?³£?°?Èº?Ä?Ì?Ó?Ù?JÞ?Ïâ?¸æ?ê?í?ï?Éñ?°ó?Wõ?Åö?ø?ù?ú?Ïú?û?Ï75½±Qw?X?ò ?á8?z?Óû;?Lõ¾ÈÊS¿]Y?Æ|"¾þÝÏ¾,?:¿H?| ¾ýf¾ÍÖP?5v¿¦uv>.P?èÈA¿ð¿^DS?ô)'?3:¿Ïaq¿Öáa½á¬\?cAb?Ö>°-¿\ÿ¿D¯9¿8&Î½Y?+Ml?Pö{?B?ÈP¶>ü8Ç½çº ¿0¯L¿bv¿öM¿Ï5l¿ID¿nË¿!A©¾Ã½í>>ª>¹?5®'?UÀE?g·\?;m?´x?ä~?ÿ?m~?z?t?~æl?*&d?ÝZ?qP?qZF?&<?Å1?ß´'?ué?³r?X[?cª?¯Çô>Íå>ä5Ö>û-È>
öº>®>îÚ¢>ïç>Þ¥>Û>ö!v>ÓXe>«U>ËG>õ^9>¡,>5¿ >u«>vX>¡¹>6ñ=rÓà=ûEÑ= ÊÂ=oMµ=¡¾¨==+=!	=û4}=À¥k=M[=*L=sî==ÞÀ0=}$=ó=ks=z='ºö< å<=ªÕ<mÕÆ<¹<ç/¬<î; <S<\Â<T <îRp<Ë£_<$P<eªA<Ô¿¿c3¾?j_G?X1?PS>®Æ-¿¶Ã`¿GÎ?°<?¬Á|¿Ýói?=¿W[/?`iH¿_s?Öfy¿Ç?©A>}yx¿Ãj?VH'?Ý S¿w¿ãA?÷SL?iª¾E¿MÃ¿uï>Í }?L<?r;;0¿%³~¿ÐÓX¿ÿóÄ¾®'5>'?Y8o?0É~?G]?Á?+º>gÛ½õcÅ¾"$¿·ÌS¿q¿ôÕ~¿«Ü}¿Ñyq¿z\¿ÉpA¿
"¿f±¿oÀ¾Ù|¾Ýû½êË´»nsÔ=.8Q>>ÇÂ>À=è>j0?`?ßÖ!?»-?»58?kA?u~I?P?à¾V?[$\? Ù`?Æòd?h?fk?pRn?äªp?d´r?#yt?v?ùVw?¬~x?y?o]z?,{?;Å{?V|?zÓ|?0@}?a}?ûï}?¬6~?ès~?ó¨~?æÖ~?²þ~?*!???ßX?Do?¨?t? ¢?®?¹?ôÂ?"Ë?8Ò?[Ø?«Ý?Eâ?Aæ?´é?±ì?Hï?ñ?vó?%õ?ö?Ü÷?óø?åù?¶ú?lû?>S]¿ùv½>xMt?©? ù?,Ûa?2>{d¿ÀÍ¾ÀÛ??-¿/¹=Hè>HÒ¾l>;¬<°q¿_Îr?Z¿÷«@¼m?:x¿x,8¿ö3?+ÓF?ìì¾Ä{¿pèI¾vI?Wo?	>^¿ ~¿ÑAI¿tÍ>¾%fì>rBd?º~?xpL?àÒ>±(½õé¾õD¿û0s¿_ß¿|æo¿ØfJ¿[è¿¦r¸¾aÿþ½úlÐ=Ú¤>ªù>_#?_B?©2Z?h{k?<þv?=}?ó?Ió~?½:{?fbu?èím?Le?ºÚ[?^äQ?¨¦G?PR=?|3?õ(?m ?X?z?m¼?¹Ðö> ç>£Ø>.æÉ>j¼>û°>mH¤>>>¹æ>8>TSx>Xeg>ØW>üÐH>	;>à..>*2">?>¶>ãå>¦µó=Üâ=J+Ó=Ä=Aò¶=qFª=Tz=­~=?E==Ém=K]=¾ñM='¨?=ö[2=°û%=w=Î¾=âÄ=-øø<T°ç<b×<¤È<¶º<À­<Ê°¡<Ow<C<ÑL<0r<<¬a<tR<mC<¿¦ ¿ÄÓm¿ñÿ¾_Ã±=d<
ñ¾õ|¿ú©æ¾õj?¯7=¾]L¿¥×?¦[s¿wZi?+t¿ñ?[¿¸?¢>û?xû¿3¾>¤M?Ï1¿86¿xB!?2ñb?¬ð<¾hùz¿ó¿i¯µ>Öîw?M?³yÕ=·6¿F{¿:c¿~Îç¾|áÜ=h?7Gi?eÈ?·µc?#?¸ì>@=½_´²¾)¿¿¹ÊN¿¥Ïn¿~¿»«~¿6s¿f_¿ëE¿Æ&¿ã¿×È¾Û¾¹d¾Â¢¼E¹=¸ÃD>?é>ó¼>­ªã> &?©?V< ?OQ,?÷6?*S@?:H?û¹O?pV?Ì[?J`?vd?xh?mAk?Ä n?ðcp?Çvr?¨Ct?¨Óu?¶.w?Á[x?Ó`y?0Cz?k{?±{?ìD|?«Ä|?[3}?C}?Yæ}?T.~?®l~?±¢~?zÑ~?ÿù~??~;?ÒU?l?_?y?H ?­?8¸?ÖÁ?+Ê?bÑ?¡×?Ý?ºá?Éå?Lé?Wì?úî?Bñ?<ó?òô?mö?¶÷?Òø?Èù?ú?Vû?ÀÖc¿7#¿óÈ>\¤@?¿=?Î<¿>³Àé¾Z}¿g>oßa?+ãg¿ä+ð>VìØ½az ½f²p½
Â´>B¿üå?½­-¿â¾DX}? Ï¾¿X¿?7`?ë¾Qß¿f«¾}¬2?\¾x?ËÍ·>Ã¿d«z¿W¿ß¾ÏÇ>GïZ?þÏ?ÎV?=«î>]v<[ÑÑ¾jÃ<¿ÁQo¿Oü¿8s¿QýO¿¬Ø¿ôtÇ¾V¾=A.>²þî>û?Îè>?W?À¨i?sØu?3ù|?ºÙ?<?Ò{?á5v?Iîn?ülf?¸]?D'S?ïH?À>?9S4?Q4*?éU ?ÈÈ?¿?ªÍ?xØø>Wíè>ÚÙ>°Ë>;2¾>ø±>µ¥>â>X'>d>cz>qi>v~Y>J>ö³<>¼/>	¥#>ö^>èÜ>>åõ=åä=Õ=÷QÆ=¸=;Î«=ç=?Ò=[= ç=Vío=^I_=NÌO=ØaA=÷3=Hz'=Û=0
=Hù=26û<Æé<Ù<¸rÊ<'e¼<<Q¯<¦%£<JÒ<)H<My<r±t<«´c<ÃåS<Ç/E<Èté>9Z¿¯§k¿è(¿Ù+¿ãxm¿JÃc¿tÍ>l}? úð¾íØ¾Rb?W~¿Áß?¿¿óo?'¿ÁÎæ<Þ<?ðôv¿Ü>j?¿ðT¿÷>[s?&W½l1q¿»U7¿r>jïn?¢Ç\?àO>éÓ
¿Ýrv¿XÛk¿¸­¿ºÀ=}?|b?ø?Qi?Ìí,?7Ëµ>Ü-<Y¾¾°@¿I¿Æk¿Æõ|¿èI¿~u¿fb¿{¦H¿*¿"
¿ß.Ñ¾»Å¾nþ¾þö½1=G8>õ;>¨Ó·>Åß>í?ô½?$? ä*? ¶5?@9??G?ÀáN?FU?çÛZ?Ìº_?fùc?«g?§âj?c®m?_p?¡8r?µt?×¤u?w?8x?GBy?´(z?vðz?{?®3|?ºµ|?i&}?}?¡Ü}?é%~?ce~?`~?Ì~?Bõ~?ý?ð7?¾R?õi?~?y??«?ë¶?¶À?2É?Ð?æÖ?iÜ?.á?Oå?ãè?üë?«î?þð?ó?¿ô?Aö?÷?±ø?¬ù?ú?Aû?OÇ½Iþ~¿_hå¾)n<=ÙÇ­=8¥¾^Ðf¿ÿÔ<¿pe'?²n?Ü¯¿­òN?ï ¿·Ý´>é Ò¾£%?$Tl¿{ýv?,±è¾uÿ¾âä~?
H¾ZÙo¿\%Æ>r?&¾~¿ÿ»î¾7¶?YR~?¤í>äPÒ¾:t¿Ãc¿3©´¾¶t >aP?{ß?1¬^?pË?A×=¹¾04¿½ôj¿¼¤¿Ø*v¿ÊLU¿U$¿4DÖ¾¦
;¾Ü+A=>ó0ä>1?ò];?;ìT?,Ãg?n¡t?ÍV|?é²?-z?p_|?ô w?ço?g?ÏA^?fT?5J?qâ??¶5?Jq+?å!?þñ?öµ?Þ?çÞú>Ùê>«Û>TÍ>}Ï¿>³>S"§>êê>¸g>é>#µ|>¢}k>ßg[>ãbL>D^>>CJ1>Ñ%>¸>>E>>Nø=îæ=ÅõÖ=ÓÈ=Í;º=þU­=ÓS¡=Í%=s½===r=9Ga=Ü¦Q=C=!5=ßø(=*?=U=®-=7tý<ºÜë<¨}Û<]AÌ<²¾<åá°<¤<E-<<É¥<´àv<½e<ÊU<xòF<ËÉ~?Rrµ½ZÝd¿º¿£¿fs¿IuÝ¾Ú,?Å¯A?ð=Y¿HJ=±?-)]¿º~o?Ui¿±C?ÒÄ¾^£¾Âd?û]¿Ý(¾½ó{?×ú²¾Pl¿*ß£>æ}?ú=§wb¿vM¿{ ê=8Áb?¡gi?9x>"{ì¾¢o¿s¿Fµ¿Ù½ü>AÝZ?Y?@¯n?é5?(JË>»X=[¾D¿éD¿:h¿u±{¿·¿7Cw¿"+e¿L¿Ýp.¿W ¿¶uÙ¾ã¾,¾@F½)Ð=Ä+>á>_®²>(sÚ>¼þ>yè?Nÿ?|u)?ºr4?E>?'F?ÙN?T?¯5Z?ß)_?'{c?=g?j?N[m?0Ôo?òùq?J×s?uu?!Ýv?þx?w#y?üz?NÙz?{?I"|?§¦|?Y}?º|}?ÔÒ}?j~?^~? ~?|Æ~?zð~?Ù?Z4?£O?Dg?»{?t?Î?ª?µ?¿?6È?°Ï?*Ö?ÅÛ? à?Õä?yè? ë?[î?¹ð?Åò?ô?ö?h÷?ø?ù?lú?+û?zôH?*¿î¥x¿¹T0¿gô¿Ë-Y¿qÊz¿Ùµd¾æât?Íå=Ù
\¿	|?`)P¿Ñ%/?N²5¿âÕ\?ó6¿ïØX?N_I¾
[3¿:-r?ÂÏÄ<Å}¿½Q>á3}?Û¡{<yHv¿[|¿ø>ý?õ'?	¾Yk¿m¿íjÝ¾q>§D?°~?Hff?(Å?c6>á¾¢+¿:f¿ËØ~¿¼x¿uSZ¿V++¿Üä¾:X¾â$¢<}õo>ZBÙ>ù?Û¾7?+R?ÖÊe?CYs?¥{?¨~?.¬?áâ|?Ãw?ÔÙp?4h?øm_?à U?MwK?_&A?ð×6?i¬,?_¼"?ú?-Ò?¸í?äü>¤Ãì>¾{Ý>
Ï>-lÁ>´>¸¨>¦@>Û§>»>å~>em>Q]>+N>p@>Ë×2>&>0>!a>ej>Cú=u÷è=ðÚØ=¦ÙÉ=à»=ºÝ®=À¢=Wy=ù=x3=Û4t=Ec=gS=3ÕD=4-7=tw*=;£=ñ =b	=9²ÿ<ëòí<ÊnÝ<Î<<Â¿<r²<]¦<?<ôÍ<EÒ<õy<Åg<b®W<)µH<ç?(ò>?W¤s¾Þ9¿àuI¿¿çM>sy?p8>ÃÙ¿TÕ?¬3>7¿Ö´:?òV4¿y}?L ½1¿K {?üª6¿.ø¥¾í?WÎ¾z¿E÷>Eø? ¹>^O¿sñ_¿û¼ÈS?IVs?£wÉ>BÁ¾ÚÒf¿æÅx¿Ûæ#¿ÓÏ½u-ß>çqR?ýë}?å2s?ôv>??^à>ü*Â=ñ9r¾­¿C\>¿R e¿Y5z¿)ó¿6ßx¿7Ïg¿$xO¿_@2¿C+¿òªá¾ãö¾O<¾½YO=Â9>6Ó>Í­>òÎÕ>üëù>??Ü\?æ(?5-3?<ÿ<?SE?F,M?"ÈS?#Y?Ä^?ßûb?ÑÎf?´"j?m?do?¹ºq?h s?üEu?Í³v?&ñw?ay?	óy?ñÁz?Ju{?¼|?r|?,}?Pq}?ðÈ}?Ù~?V~?~?êÀ~?§ë~?¬?¼0?L?d?ay?k?
?¨?I´?o¾?8Ç?ÔÎ?kÕ? Û?à?Yä?è?Cë?î?sð?ò?Wô?çõ?A÷?mø?qù?Rú?û?r?ÕM>TO¿#ÿ¿ä|¿«}¿¶¬(¿4u½>µôv?#ð¾=¿û>v? ëz¿;<i?þj¿G^{?Êx¿Á(?àJ=O[¿çW?x>è¡¿[Úx<Éâ?X,8>,§h¿õ2¿öRº>Ø·}?R'??×S¾¢`¿u¿íA¿8µ >	Ò7?bC|?º'm?£6?Â;>8¾B!¿ÅÊ`¿Û}¿:ëz¿_¿¾1¿l:ó¾Ëïu¾M´ø»V>z4Î>L\?î4?0VO?æ¿c?	 r?äz?÷<?Ò?Y\}?É}x?ïÄq?æªi?.`?×V?¶L?gB?ã8?«å-?Sí#?¹@?bí?ü?Îçþ>-­î>
Kß>ÀÐ>KÃ>0¶>½ú©>>Àç>úæ>×>ço>:_>ôO>y²A>7e4> ý'>²k>(£>x>«rü=P ë=ÀÚ=oË=:½=qe°==-¤=ÝÌ=5=¯Y=Xv=åBe=ï[U=ÞF=EÈ8=ö+=J =Pì=x
=ø =	ð<ë_ß<¤ÞÏ<ÅpÁ<8´<7§<:ã<Ù<Áþ<7?{<úÍi<±Y<ÙwJ<Ð¦¾GÐz?¦ï?'¨»s¾ Ë>@?Ôm?´ã¾#ûa¿×Z?ûû¾.K¾Ó>ëÏ¾èA>¾Kq>A¿SC?¿	¿­_x?Ui[=qø¿®ô¼KÓ{?3Õ>9|7¿\rn¿Øj¾ñA?Fvz?¹s÷>æ]¾:b\¿àÓ|¿,2¿R.¾ÎÀ>àAI?Ù°{?w?¸F?güô>Ýø>°K¾L<û¾Wo8¿ÍDa¿Æx¿þ¿ÖQz¿íQj¿bºR¿£ü5¿(¿éÍé¾	ÿ¦¾K¾.â½ò=:©>I0v>T¨>D%Ñ>²Àõ>P5?Ô·?æ&?tå1?)ß;?¦¡D?OL?¹S?EåX?^?{b?_f?Ái?
³l?ûAo?özq?is?ñu?v?ÿÌw?åx?Ú×y?`ªz?Ý`{?ÿ{?|?áþ|?Ëe}?õ¾}?4~?O~?~?L»~?Êæ~?u?-?WI?Ða?w?]?C?§?ô²?G½?8Æ?÷Í?«Ô?zÚ?ß?Üã?¢ç?æê?ºí?-ð?Lò?"ô?¹õ?÷?Kø?Tù?9ú?ÿú?]Wr>Åg?ºÕ¡½õ92¿Í(T¿BM+¿\&½4U?°÷,?¿Y¿öcw½Á¿>?¶y¿iÝ?Ç©¿Ü&}?ó±Y¿[§Ñ>pp¯>s®u¿21?{Ãå>ww¿>33¾£z?2Ë­>JU¿¯äJ¿Ûr>»w?CÁ<?FUÎ½å¤R¿eÙz¿}Ô¿È=ô)?Kx??ér?C*?·as>ÊaX¾Ð¸¿'[¿}å{¿}·|¿c¿«³7¿ý¬ ¿ì¾9½ðî<>ìÃ>á¬?E0?ömL?¢a?Øp?Þz?Üí~?Dí?ÔË}?}/y?ä¨r?³j?i·a?
X?ñM?à¥C?S9?/?¾%?8f??
	? u ?¡ð>uá>¸tÒ>Õ£Ä>U¡·>bf«>4ë>f'>6>½¢>& q>à"a>}¼Q>`\C>ò5>¦o)>"Å> å>Â>½¡þ=	í="¥Ü=.aÍ=ä)¿=!í±=ê¥=^ =¦q=ä=O|x=¶@g=t6W=HH=Tc:=t-=Wk!=®7=ÛÊ==Kò<Qá<G­Ñ<NÃ<àµ<ù¨<4><¾S<=+<wn}<iÖk<ÿv[<:L<Dºx¿smÙ>3?Å7?mC?½=>?¶_?ââ?Zº<¿® ¿[?o¼*¿Ïa>¡=
R½S¾é³?¿i¿p?Æå¾úl8¿Æd?v>­|¿Ö%S¾õÌp?;?$¿_·x¿~¾%ð,?²~?n{?aL¾ÊJP¿Å=¿óq?¿At¾¡>ÃU??ªx?!8z?ø1N?æ?6>ky#¾ËÇê¾çK2¿O]¿v¿ì×¿Ù{¿é²l¿\âU¿B¥9¿¼¿óÝñ¾yû®¾üZ¾ »½3ÝÃ<é>¿±j>m£>;vÌ>ôñ>´W	???%?{0?½:?#£C?(pK?ÕCR?;X?p]?6úa?îe?_i?Û]l?ö÷n?«:q?;1s?åt?`v?¨w?hÅx?o¼y?z?CL{?-í{?¡x|?yñ|?-Z}?å´}?|~?G~?~?¡µ~?áá~?5?g)?'F?_?t?K?x?z¥?±?¼?6Å?Í?êÓ?ÒÙ?ðÞ?_ã?5ç?ê?hí?æï?ò?íó?õ?òö?)ø?6ù?ú?éú?0¿Q£a?°H5?Fm=Àmf¾HF½ä+?´ö?SÕ6>º×¿ªÒÓ>z~Á>ÛÀL¿ o?A~s¿fûa?ª%¿ÌQ>}?eâ¿Þ@?þ+"?æd¿ý¶·¾ï5m?¥Àú>ÄS=¿_¿³2Ø=þm?lO?MUÖ;[C¿~¿I&¿Uj»x"?g¿s? ¤w?ðR5?Ys>ß»#¾H¿mÈT¿w¿y¿$ ~¿ò¡g¿K§=¿Ô¿vý¾A½n:#>JÁ·>íë?/l,? rI?âr_?Éo?v4y?Z~?Vü?N1~?¯Øy?¬s?P¶k?£Ôb?y8Y?)O?iáD?á:?R0?J&?t?¾ ?¤
?­u?ü|ò>ýæâ>²(Ô>Ê>Æ>$¹>¥Ñ¬>@¡>Ìf>?=>yº>#«s>uc>°S>#E>¼7>â*>>	'>|î	>^h >×ï=(Þ=ã$Ï=ÎÀ=Êt³=§=Ûs=±­=¦= z=>i=öY=,J=aþ;=,ó.=dÏ"==>ÿ=6=y5ô<+Bã<é{Ó<ÖÍÄ<$·<ìmª<.<£<¹W<¸<ØÞm<N[]<9ýM<o^9¿ Ûñ¾Á4?Ö?'oy?U×?û6V?ø¼êâ{¿ã½Yi?Êm¿©?LT´¾¬>(ð¾#¹C?ë}¿ÑO?>3ö¼`ù\¿ßF?hEå>Ìón¿À¾_2_?¶O,?û¾Ï~¿æÄ¾¾ 	?þ?àÈ%?GÜ½½£B¿ÿ¿Ö£K¿¾À>Õ¶4?ëÙt?Í´|?VU?ìU?í`>la÷½&Ú¾Âó+¿Ç!Y¿Ëut¿¿	º|¿Ññn¿¬ïX¿Ö9=¿ø¿iÚù¾¥ë¶¾']j¾
UØ½0G-<îò=2+_>îå>öÁÇ>ÜYí>vw?"f?¹ #?OO/?ñ9?Ì¢B?J?xQ?W?tÚ\?Öwa?û|e?Èüh?ùl?S­n?×ùp?ñør?¤´t?²5v?Ãw?¥x?È y?¢zz?|7{?+Û{?i|?óã|?vN}?¾ª}?±ú}?ó?~?ò{~?ê¯~?îÜ~?ë?°%?ïB?D\?4r?4?¨?é£?@°?ðº?1Ä?5Ì?&Ó?)Ù?^Þ?àâ?Çæ?(ê?í?ï?Ññ?¸ó?]õ?Êö?ø?ù?ú?Òú?8a{¿>>£#?Ëæ>?5³ÿ>³{?Ï,q?ë.P?¥A¿¾Ìb¿c©J?ÏF½t×÷¾ç:?J6G¿h÷,?fì¿¾Vß)¾H?ry¿	s>¾I?Ã¶H¿¿öpX?n ?)¨ ¿®¿n¿Püá¼ëî_?T_?`øè=Ü1¿Íø¿6¿ï'¬½.q?Å²m?ÁT{?'è??á¼°>fFÝ½Üó¿ÛN¿Ã'w¿$¿9uk¿ÜbC¿õg¿NO¦¾Ë·½Éj	>5_¬>?.(?udF?'1]?÷m?îEx?v'~?Àÿ?Â~?Zyz?B[t?û²l?Õìc?²bZ?^P?F?åÅ;? 1?ñv'?m­?á8?ó#?u?<cô> ³ä>òÛÕ>*ÙÇ>5¦º><®>¢>ò¥>h>Ò>Ýµu>Ôód>·LU>Ã¯F>Ô9>mT,>Êw >ãh>l>Ó>ñ=!oà=èÐ= sÂ=lü´=4s¨=TÇ=·é=FÌ=±Ã|=L<k=vëZ=Ï»K=l==»q0=o3$=gÎ= 3=U=§Kö<J3å<JÕ<^|Æ<0µ¸<Æâ«<'ô<Ù<4<|æ<Fço<?_<é¿O<%«A>81}¿Î§½Ç*?þÉ]?IÔI?S´«>wû¿êwm¿*ð>i?ýr¿`?<ð.¿õÇ ?º<¿vUm?ðs|¿á¥?;d>'u¿åÇ?úæ?à'Y¿Xµ¿ª~G?ïOG?BÅ¸¾ç¿«ø¾òFú>V~?Cn8?ñÞr¼D3¿¿³°V¿Q¾¾F­B>o)?ªCp?[~?¯ø[?"Ô?k>n§½ôÈ¾Êh%¿[¼T¿Hr¿Fø~¿6¯}¿Tq¿ñá[¿ý¹@¿Ê!¿Sá ¿þÎ¾¾¯y¾.~õ½ªÏ4»Ù­Ù=S>Ä§>Ã>é>?¹?%"?ó .?Òr8?£ A?r­I?£¹P?ÉâV?°C\?oô`?
e?8h?e±k?bn?z¸p?0Àr?at?ó
v?®^w?[x?æy?vbz?"{?É{?IY|?PÖ|?¥B}? }?Óñ}?D8~?Ju~?&ª~?ï×~?ÿ~?ñ!?±??uY?Åo??Õ?T¢?á®?À¹?*Ã?RË?aÒ?~Ø?ÊÝ?`â?Xæ?Èé?Ãì?Wï?ñ?ó?.õ?¡ö?ã÷?ùø?êù?»ú?®(¾¾v4¿T???î·t?<[z? t?T­>TO¿¢Ò¿£K|?#Ûý¾¯«½àÓ>Ô¨ ¿@VÈ>#,½Üºã¾©Ök?+c¿Æa=~>g?9$¿Y.¿kØ<?Î??./ ¿Æ~y¿$¾/ÛN?éBl?®í`><>¿7¿cWE¿ï'¾îõ>ï}f?¨õ}?ÊI?`wË>îód½ê-ð¾òG¿t¿Ä¿"øn¿«äH¿¢¿I|´¾ÒSï½ß=Oä >oü>ý$?¾CC?ÝZ?~òk?SHw?6°}?÷?.Þ~?w{?)u?©m?ûÿd?º[? Q?öOG?û<?Ë·2?³¡(?Ï?úO?i/?¹t?`Hö>\æ>w×>òrÉ>î'¼>§¯>»è£>×ä>µ>té>RÀw>ûÛf>W>@YH>Î:>­Æ->Ñ!>®ª>OF>?> #ó=Tâ=.¬Ò=±Ä=¶=Ñß©=É=»%=sò=[ç~=:m=òÅ\=puM=u4?=Ið1=y%=Ã=h=t=Óaø<h$ç<+×<å*È<×Eº<W­< O¡<l<¯°<þ<´ïq<ê#a<Q<L°m?ê5¿dïO¿jÐ½_Q>FÔU><ë¾Sâp¿ê¿ÿX?	->gQ^¿!?íi¿R]?×k¿h?~He¿ )Ç>ð?ì>ó¿Ý§Û>Ú}D?t;¿ÜÖ,¿W*?]?^e¾±|¿ÊÑ¿1%Å>n¿y?ç;I?Ì =\#¿«|¿¤`¿®ËÞ¾K>Ú?ôêj??Éb?!?õ#>&s.½o ·¾ì¬¿| P¿o¿û>~¿7z~¿$s¿Í¸^¿V%D¿%¿Ë¿ø¤Æ¾gy¾5M	¾{Ö¼}dÀ=¢H>e>4J¾>ûÝä>6¯?p
? ¨ ?k°,?µJ7?ª@?¤ÉH?WòO?­4V?Ä«[?p`?;d?Ü4h?Zk?:n?vp?ør?¶Qt?Úßu?K9w?îdx?Èhy?Jz?f{?±¶{?jI|?È|?»6}?/}?áè}?0~?n~?V¤~?æÒ~?;û~?*?k<?V?Qm?ù?þ?¼ ?­?¸?!Â?lÊ?Ñ?Ò×?5Ý?ßá?èå?hé?oì?ï?Tñ?Kó?ÿô?yö?À÷?Ûø?Ðù?¤ú?ï¢?ek}¿óv¾"?Mf?.b??>¾Ñ|¿òA½Fs?sS¿µÓ©>Ö¦	=$¾$ ~=E{>ôx/¿	û}?¡>¿~9¾'­y?Ý(ñ¾9ÇN¿K?ÝX?![¹¾ ¿K¾:?¨v?pc¥>Ø\	¿í@|¿¨´R¿,´x¾ÂÓ>à)^??MïR?3å>©k»UâÙ¾m?¿>¨p¿¡ÿ¿x)r¿+N¿&¿AÂ¾ôv¾«=@R>¡ò>
r ?Æ@?*xX?{Ej?µ;v? +}?ã?%?¡{?¼ðu?n?f?ª\?¸¼R?ñH?ä.>?ç3?áÊ)?ï?f?:?7s?e,ø>/Jè>A@Ù>"Ë>,©½> ±><¥>|#>!½>² >Êy>ìÃh>?ÜX>J>¬&<>Ö8/>'*#>jì>'r>¡®>¬+õ=î8ä=ÄoÔ=:¼Å=¸=iL«=9n=»a===Ô7o=l ^=/O=}Ï@=Ön3=û&=e=b==þwú<é<ÊçØ<kÙÉ<~Ö»<xÌ®<ª¢<P_<*Ý<¼<"øs<8c<IES<*nP?è>xx¿Ñ¶E¿§ß¾~Ðï¾=Q¿:ww¿d½¢Õ?ul¾%O¿Ôq?ûÚ¿:¬|?@¿g'x?¶f:¿{] >ë+?ÌÃ{¿s1b>öÒa?'í¿ØK¿"?¡¢n?ý¨½lþt¿rC/¿+c>©Fr?ÐX?r.>d¿³Ux¿i¿ieþ¾FX{=w?Ôd?ÿ?ª©g?SÜ)?&®>¦H\»¦¾'Â¿UOK¿ÐÎl¿âT}¿ë¿÷Þt¿æsa¿{G¿÷@)¿ñ©¿mÎ¾ü¾Ô¾ ñ¼§=ik<>>õ¹>bà>FÇ?éX?Y(?½]+? 6?ä??5äG?)O?EU?±[?ê_? #d?µÏg?%k?ÃÉm?'4p?HMr?£t?f´u?w?<Dx?nLy?1z?øz?:¤{?i9|?±º|?·*}?Æ}?Üß}?µ(~?Ðg~?y~?ÒÍ~?Õö~?[?9?ÄS?×j?Ô~?#? ?¬?Z·?Á?É?ÒÐ?%×?Ü?\á?xå?é?ì?Åî?ñ?ó?Ðô?Oö?÷?¼ø?µù?ú?[²?½ô¾3e¿'ð½ë©¸> À>Ï½êB¿ÐÆ`¿Âï>±1?I}¿dS0?ÒÊ³¾Ã8O>R¾qò?yþ]¿2}?s¿¿pÐ¾Ýÿ?ýþ¾(®g¿é¯é>Jl?Áé[¾U¿~WÔ¾3x#?a|?gØ>18æ¾}w¿­}^¿Äí£¾°>øÀT?ÿ?tO[?=éþ>G=Ã¾=£7¿bÃl¿ÒÕ¿%u¿4S¿Øê!¿[Ð¾°'/¾n=²ª>rè>ÅP?ÙË<?JV?h?% u?»|?Ä?Þb?ø'|?°v?o?g?È]?ÏæS?	³I?×_??U5?yò*?¤!?{?ÆC?q?Hú>ê>LñÚ>¸¤Ì>î)¿>×z²>-¦>ßa>Yç>Å>oÔ{>¥«j>À£Z>Í«K>m³=>èª0>9$>.>ò>ùÅ>)4÷=Àæ=O3Ö=»`Ç=,¹=ü¸¬=¥Á =·=Ä>=P=5q=âz`=«èP=jB=`í4=_(=v°=ÁÐ=²=(ü<¡ë<j¶Ú<ñË<$g½<QA°<¤<4¢<¥	<[-< v<ìd<øU<~SG½Cä`?Pä¾Þ;~¿÷Än¿T6m¿×¿de&¿ê
õ>i2b?I8¿Õ­¾"9?s²o¿´z?¹2v¿SJX?Éüþ¾ò'¾1ëU?6Ói¿²;Âªu?9ÏÙ¾Çc¿UÿÅ>Ûz?¤qx= òh¿E¿~'>Zh?(©d?jÆ>Ë?ý¾r¿ëbp¿a¿í½g»?^?1²?ï°l?Ó\2?ß¸Â>~ï=C¾ª¿$JF¿Øi¿(:|¿7¿v¿ãd¿&¼J¿ä,¿Ð}¿&Ö¾¤¾S&¾F/½¼=ÂÈ0>ÐÒ>õ¾´>ÎMÜ>±¹ÿ>÷¤?G¦?ì*?ô4?T>?(ýF?a_N?ÔT?vxZ?d_?à­c?Ãig?{©j?°|m?1ño?"r?(ís?u?ív?F#x?Ù/y?¹z?âz?{?F)|?µ¬|?}?F}?ÄÖ}?Ô ~?ý`~?~?´È~?fò~??Ë5?âP?Yh?«|?D??´ª?#¶?	À?È?Ð?vÖ?Ü?Ùà?å?¤è?Åë?{î?Ôð?Ýò? ô?&ö?x÷?ø?ù?vú?UWÿ>;Ö>\k¿<2L¿W¾¾WS¾Íé!¿Ý¢~¿X­ù¾;àX?>îþs¿"Øl?º.¿,ë	?å5¿XkE?ny¿¸i?OP¤¾æ;¿èy?L £½Ñ%x¿Åj>v*y?m£½"±z¿:3¿ÅÓ	?å±?Kv?Ôi·¾¥Òo¿Øh¿qÊ¾û>òNJ?kf?¥âb?}»?o Î=HÌ«¾^/¿Ârh¿*G¿/w¿¢ÿW¿(¿ÇÞ¾Ì¶J¾Ë=§Þ{>FjÞ>?Gu9?yS?Oºf?³õs?ú{?æ??P¦|? iw?Ñhp?Ýh?rá^?aU?9àJ?g@?-A6?x,?s,"?õ?©L?n?ñû>Ýë>¡Ü>³<Î>2ªÀ>(ä³>lã§>  >[>­.>Þ}>&l>k\>ÝTM>@?>â2>9Ü%>´o>°É>GÝ><ù=è=Ðö×=3É=´»=%®=¢=°Ù=èd=©=M3s=UUb=D¢R=D=êk6=Ã)=Îû= =Ñ=Q¤þ<¼÷ì<Ü<w6Í<Ê÷¾<)¶±<
`¥<å< 6<ûD<þx<ÔÐf<§ÊV<sä]¿Ånh?¶gÉ>Ùf¿ç¦m¿cs¿þJF¿ÝÓ½z_??³v¿ò>ÕSÂ>1;¿¬W?Á»P¿4ú"?.f¾Ö´Ñ¾qur?<ÌJ¿^¾Ç/?O¬{¾:Ût¿k>=? qO>ïÂX¿»W¿2êG=Ô[?o?Â³>ÐÕ¾#k¿rNv¿ Þ¿1 ½E$í>^V?±~?(q?:?öÖ>Î´=uE¾#h	¿0A¿j­f¿ïz¿Ý¿ "x¿rf¿çæM¿x0¿OF¿(ÑÝ¾.,¾2É4¾ªe½Î½h= %>>Vò¯>Xþ×>íßû>¢î?ð!?ü±(?Æ3?ü=?F?ºM?"T?ÝY?Ü^?Ù7c?g?Pj?/m?³­o?Øq?Eºs?n\u?GÇv?x?y?½ÿy?óÌz?Ö~{?|?|?b}?±v}?Í}?ã~?Z~?~?Ã~?íí~?¤?p2?úM?Õe?}z?a?ß?J©?é´?ù¾?°Ç?<Ï?ÅÕ?nÛ?Uà?ä?@è?oë?1î?ð?¥ò?oô?üõ?T÷?}ø?ù?^ú?xxë¾~z?Ø¾WË|¿îg¿ÏX¿Ýx¿ç`¿ÂX=}Ó?½3R¾ms9¿ Ý?ÿh¿õ×M?FOQ¿IPn?ôk¿ô{D?½ÏG¿±g?K >»¿Ý¼ñ=àp?6»=K·p¿Ï#¿d!Ü>|n?1?QÆ¾vf¿¾ûp¿¤ï¾5L>Òà>?¢¹}?Í¡i?D?Âu>È¾¤½&¿S¸c¿ãS~¿¶Éy¿Ö\¿P .¿;ë¾ÿf¾äë;§Cd>`/Ô>Ü?^6?ÄßP?fÜd?r¼r?"N{?b?J¿?	}?]x?òFq?i?}ö_?g0V?~
L?ºA?k7?Û<-?óH#?Ò¡?®T?}j?¢Ñý>$¥í>#QÞ>ÔÏ>ø)Â>Mµ>Y6©>ÞÝ>';>jE>vç>nzn>92^>ÇýN>Ì@>Ä3>$5'>A±>bõ>ô>ñDû=?çé=GºÙ=£©Ê=4¢¼=¯=oh£=¥=	=êº=1u=Æ/d=Ü[T= E=qê7='+=%G=~9=ð	==] =×èî<¦SÞ<üäÎ<oÀ<+³<»¦<ü'<b<\<kz<!µh<WX<îQc¿Ù:S>t?~!>5Ú¾!¿p¾íõ>+¤?ùï=\z¿ÿy0?¡¦½Ô¾Ø2?Ge¿»>¯=$¿ÉV?sk ¿´ÀÙ¾û}??Bj½Ü5~¿6G=í~?C®>lºD¿!g¿³f½RK?¸øv?MêÞ>
Ç¬¾Z:b¿×z¿v*¿5¾èMÑ>ON?~þ|?Éu?L@B?jëé>Fµé=º6`¾!ý¿Ð¨;¿©Oc¿²sy¿Oþ¿üy¿Cûh¿nûP¿û3¿ ¿,lå¾àª¢¾Ý5C¾¶½fú5=¾q>º`|>5!«>ªÓ>Tø>ð5?Z?òX'?2?ßz<?;*E?¢ÆL?GoS?@Y?T^?íÀb?f?öi?·àl?¬io?pq?ûs?ë/u?¨ v?àw?ûõx?æy?·z?ék{?|?f|?}?l}?ZÄ}?á~?+S~?~?V¾~?ké~?¼?/?K?Kc?Jx?z?9?Ý§?¬³?ç½?ÃÆ?nÎ?Õ?ÔÚ?Ïß? ä?Üç?ë?æí?Sð?mò??ô?Òõ?/÷?^ø?dù?Fú?Xå~¿ý??(	?í¿)Îs¿2}¿mj¿âÎå¾ÜÂ?úMb?x¯#¿Öð²¾°f?yØ¿^v?Exv¿?4to¿¼È?÷L:>g¿W>J?a¦>µ}¿T¢s½£¼~?:èx>*Îa¿T<¿8¡>Ä{?"0?$)¾á[¿Ow¿h¤	¿)Æ >Ö2?ûz?o?`ì"?¤-Q>Å*x¾Å¿>^¿\ü|¿úª{¿ÙÔ`¿ôø3¿zÇø¾­¾êÝ¼ËL>ÚÉ>?q2?5N?qîb?utq?z?£?aÞ?}?DÄx?ær?j?<a?ÞOW?Ó1M?PäB?ó8?_.?d$?³?Ó[?*f?±ÿ>Blï>ìÿß>ÔjÑ>?©Ã>µ¶>òª>y>½d>û[>Hø>}ap>1ù_>¦P>ýXB> 5>ü(>¿ò>!>Ä>=Mý=êËë=²}Û=
NÌ=®)¾=þ°=Î»¤=Q='±=´Ì=·.w=3
f=qV=;G=÷h9=,={ =Ûm==Ph=ðÙð<C"à<Ð<Â<Ù´<ú¨<ßj<<9t<Ø|<nj<PZ<s¾½HW)¿(X?EVR?ÿ#>>\Í>¢Ãd?@Q?ßYï¾'ÕD¿7Ûo?Ãà¾ =½¯N>ef¾Òl=lµ>àS¿@º{?' Ú¾ió¿9r?­>÷¿é1Ë½pRx?Ó:ñ>§3-¿/úr¿¯x9¾Æ9?w|?d?r¾>ÙW¿¹÷}¿W|7¿$ÈI¾î³´>ÁvE?z?J^x?ìI?déü>Ð¬>;¾d×ô¾g6¿¿_¿Èw¿õ¿l×z¿	Dk¿fùS¿(m7¿÷³¿÷ì¾³ª¾FQ¾êQ©½Ò/=-¾>î²q>³K¦>'QÏ>øô>êz
??Ñý%?¸d1?þm;?_>D?øK?¶ºR?æ¢X?Ê]?Ib?23f?Ri?Òl?%o?äaq?ISs?u?»yv?Ç¾w?²Øx?)Íy?¡z?ÕX{?ø{?|?§ù|?Da}?	»}?Î~?-L~?~?¹~?àä~?Í
?¦+?H?¼`?v???l¦?m²?Ó¼?ÓÅ?Í?`Ô?9Ú?Iß?«ã?wç?Áê?í?ð?4ò?ô?§õ?
÷?>ø?Hù?/ú?Ô´¿éì¨½7}?Q¾J>eú¾Ñ+¿/^ù¾b³>wl?u6?òÃl¿Ä=êð#?2Ìo¿°à~?Þp¿|}w?DæJ¿Â¤>«Ù>Ö¾z¿ü"?lÏ?ìr¿ÐÃp¾®w?(Ç>DN¿&ÞQ¿ë©G>¿t?EC?ÆÓ½×ÛM¿ðC|¿¿G7Q=eJ%?-w?t? Ã-?u©>XG¾¨y¿ØY¿A{¿V6}¿gÜd¿£9¿Jë¿Ò2¾2B3½~®4>k¿>'?Ò
/?zK?ð`?Ïp?²Íy?Ñ~?bó?í}?Ðfy?¨ðr?pk?¨b?ÁkX?4VN?¡D?Ú¸9?Â/?ø}%?PÄ?b?a	?¯Ç ?p2ñ>ñ­á>÷ Ó>(Å>³¸>7Û«>ÐX >>_r>±ü>RHr>û¿a>.OR>FåC>?r6>Áæ)>-4> L>ô"	>wUÿ=°í=AÝ=iòÍ=!±¿=k²=(¦==B×={Þ=f,y=äg=ÏW=ÖH=|ç:=ï-=ÐÝ!=7¢=.=cs=	Ëò<àðá<BÒ<¹©Ã<°¶<òp©<Â­<»<Ø<E"~<»}l<´\<©§I?¯ ¿:J>¡îz?L_?·Æ=?_?}?ý'Ä>èÀX¿!¶Â¾pÑ~?¨D¿JA³>¿½íG=ôé¾?ir¿Ûg?ðgN¾ÉiE¿[\?ax£>ÁÒx¿t¾ÁÖk?J ?¿!{¿Ç"¾Ð%?µo?-?¶E.¾L¿tª¿|C¿u@¾q>§þ;?w?{?P? º?;AD>Bµ¾(lå¾dG0¿Éý[¿»íu¿5Á¿Àû{¿zom¿{àV¿&Î:¿X¿lqô¾1±¾©ï_¾jÄ½"À <Æ>üýf>ïq¡>óÊ>ì5ð>½?? $?í00?]_:?ìPC?%(K?ÜR?X?@]?hÐa?Êe?â?i?RBl?àn?ã%q?/s?ÕÕt?Rv?¾w?/»x?³y?ìz?E{?gç{?¡s|?#í|?lV}?¤±}?« ~? E~?n~?Ì³~?Kà~?Õ?6(?E?(^?Øs? ?ã?ù¤?,±?¼»?âÄ?ÏÌ?«Ó?Ù?ÁÞ?6ã?ç?iê?Ní?Ïï?ûñ?Üó?|õ?åö?ø?,ù?ú?ô¨>u7Y¿áY??K X?ÖMr>^½>ü4-?pX|?÷=ßÌ~¿ï¦?­3>GK;¿7e?h2k¿ÇÜV?¯æ¿¥ÀZ=x#?ºÙ¿z¤ç>ï,?,©^¿Î¾¨h?0?[6¿ü~c¿A¡=wj?ù³S?R¸	=ÄÓ>¿¿ú~*¿C®Ã¼úA?cUr?ý«x?>8?`>Ôb¾)à
¿¦$S¿Õ"y¿Ck~¿N h¿?¿ÑU	¿Ö¾Ñ»½.»>%å´>v¶?Öp+?9¯H?óâ^?¸n?Âùx?ìw~?Kþ?NI~?ýz?1¼s?ök?¾c?Y?wO?0E?ÀÜ:?A 0?y&?ìÓ?ug?[[
?=¶?©÷ò>0[ã>{Ô>K¦Æ>g¹>(-­>ã¡>E·>>ö >î.t>c>¨÷S>qqE>Øã7>r?+>u>,x>:
>Ñ® >ï=iß=¾Ï=8Á=×³=}b§=oÉ=[ý=?ð=*{=¿i=Y=qJ=þe<=S/=$)#=Ö=ûL=u~=!¼ô<{¿ã<ðÓ<]:Å<·<éËª<¥ð<è<w£<Y<bn<cÕ]<ò¨q?y¿ *¿½4	?Ùºx?ÖÐ?x}?<?Dh,¾KÑ¿±
Æ=}WZ?bv¿ç.?©Åâ¾",Ê>:-¿É=P?y¢¿9E?Â÷=Md¿;Â<?³¦ü>@j¿úÕ¾/ÆY?n3?-Êê¾U¿?Í¾Øï?òÚ?¦*?Õ±¬½±ö>¿Mí¿5N¿
¥¾cAs>eî1?vÄs?f9}?W?ÆÁ?Kk>!Bã½½Õ¾AR*¿HX¿Âãs¿ßb¿Îû|¿S}o¿_°Y¿­>¿ð¿Úû¾èé¸¾G;n¾÷¯ß½+iì;Ýì=6B\>	>Æ>CIì>þý?Sú?`A#?:û.?ýN9?äaB?ÃVJ?½MQ?'dW?´\?ÏVa?9`e?Âãh?7òk?jn?jép?®êr?C¨t?ô*v?qzw?ox?Çy?tz?82{?Ö{?e|?à|?~K}?,¨}?wø}?>~?Fz~?w®~?¬Û~?Õ?¿$?B?[?q?­?3?£?è¯?¤º?ïÃ?üË?õÒ?þØ?8Þ?¿â?«æ?ê?í?ï?Áñ?ªó?Qõ?Àö?ý÷?ù?þù?Iþx?ogn¿÷±¼¦x?ÚU?ï?fÜ5?»£y?>?¯ñî¾ÜU¿ÁW?b¾2¢Ô¾å-?<¿Gí ?Ñ3¤¾à_¾QÚO?¢¢v¿x)|>jÊO?7B¿á¿8ãS?¾b&?ôü¿³q¿%­T½o]?oëa?ûm>ö$.¿Øÿ¿P9¿`,Ê½}?­wl?ä{?æÃA?ú©µ>ÎÉ½¬ý ¿XÚL¿Y¢v¿YI¿rl¿¸hD¿ç¡¿ûè¨¾ù¾Á½O±>9Hª>à6?ÓÆ'?ÓE?±Å\?ÞDm?»x?®~?ÿ?a~?Æz?}t?zàl?wd?·Z?P?ëRF?¢þ;?¾1?¡­'?oâ?ïk?ÝT?4¤?î»ô>ªå>]+Ö>$È>²ìº>Ä~®>±Ò¢>6à>£>>Ov>Me>ýU>}ýF>XU9>,>Ù¶ >«£>3Q>Ý²>yñ=´Çà=;Ñ=ñ¿Â=ôCµ=Íµ¨=U=p#==¹'}=hk= B[=yL=ä==·0=wt$=í
=ôk==8­ö<å<
Õ< ËÆ<^þ¸<à&¬<3 <<»<<TFp<_<ïm>Üº>ð¿as¾|¹?ZAJ?,4?×b>Ö+¿~ib¿b?XÊ	?à/}¿áh?Àß;¿p³-?%G¿Ezr?Îy¿iq?77>àx¿?'Q&?çJT¿Cª¿.B?eÀK?ò5¬¾§¿j¿Íð>½·}?¤;?yb;0¿ä¿~¿wX¿'Ä¾H½6>#N'?Xo?Á~?  ]??B9>,Ì½ÞÎÅ¾1$¿âèS¿øªq¿Ú~¿q×}¿Pmq¿Âh\¿o[A¿{"¿¿f>À¾_z|¾UÐú½7ª»Õ=íQ> ²>%+Â>Xè>&<?øj?à!?¥Ã-?â<8?IqA?öI?XP?ÃV?(\?SÜ`?õd?òh?¡k?ETn?{¬p?Æµr?Vzt?v?àWw?tx?Èy?^z?¯{?¬Å{?fV|?ÏÓ|?z@}?¡}?2ð}?Û6~?t~?©~?×~?Ìþ~?A!???ñX?So?¶??
¢?¡®?¹?úÂ?(Ë?=Ò?_Ø?¯Ý?Hâ?Dæ?·é?³ì?Jï?ñ?xó?&õ?ö?Ý÷?ôø?æù?í8?Æ;¾ÆF¿¿A ?f|?é5z?kË}? ¦m?	ý>¡X¿fxõ¾ªM~?)
¿Ó½*¬¾>Íî¾¶>~â½áò¾èÏn?_¿tyæ<kêi?r{¿Û\2¿ÒS9?5BB?Uø¾dz¿ÝU3¾·L?ªm?ém>aô¿öý~¿BîF¿$%1¾,ò>Çe?0~?¼ÜJ?tÎ>ý·L½Ä®í¾Ì2F¿¦Às¿JÐ¿ÊXo¿I¿&Î¿³¶¾í¡õ½«(Ù=>wRû>!$?íçB?õZ?ÀÂk?ª*w?Þ¡}?Øõ?Äæ~?&"{?@u?!Åm?Ðe?À©[?s±Q?ÝrG?|=?HÚ2?mÃ(?×ï?o?£M??;ö>Z³æ>¿×>N¡É>S¼>
Ð¯>:¤>î>´>	>uûw>=g>+HW>jH>¿Æ:>ð->ø!>Ï>Ch>â¶>^ó=ôâ=OßÒ=OGÄ=_°¶=	ª=8A=I=Á=\%=Ésm=«û\=p§M=ÿb?=2=É¿%=G?=ì==Nø<±\ç<M×<£[È<4sº<Ö­<iv¡<û@<µÒ<Å<¡*r<ÀZa<a1¿ªèv?¦!¿Õ]¿+¾ÕX>`> Y¾¾Ô>v¿Sh¿ñ¨`?eë=W¿âÕ?õm¿sb?Do¿Þ?å\a¿Lq¸>ZAù>æ¿|Ð>ß@H?5£7¿00¿j»&?Ì_?Ñ%U¾#|¿úµ¿Xó¾>y?AK?sµ=!!¿ê#|¿×a¿Knâ¾Ëó=y&?ëCj?®?h¾b?j"?Gx>D½)¦µ¾Àæ¿O¿ÉCo¿ì&~¿~¿6?s¿[	_¿D¿ù%¿¿;¿8Ç¾V¾¤ò
¾_4¼½=s·F>VÌ>fÀ½>jbä>x?zÙ?Ê| ?/,?)7?@?¿¯H?¯ÛO?â V?~[?ó``?d?r)h?1Pk?n?op?vr?Lt?óÚu?
5w?>ax?ey?PGz?ÿ
{?´{?G|?ÿÆ|?`5}?}?Ýç}?£/~?Ñm~?­£~?TÒ~?¼ú~?¼?<?MV?
m?»?È? ?X­?l¸?Â?RÊ?Ñ?¿×?$Ý?Ðá?Üå?]é?eì?ï?Lñ?Eó?úô?tö?¼÷?×ø?Íù?äF¾o?a»{¿¾Ö?b?i^?»|?,&¾
Ï¿í­U¼¹q?EV¿³·²>.<`¾éÂA=¦>*Ü1¿áX~?y<¿ÍðC¾6z?í¾µP¿+´?¶Y?"¶¾)A¿/_¾}¤9?_fv?»ª§>Qj¿Ý|¿lAS¿15|¾Ò>Æ]?«?cQS?­æ>ÅòªºBåØ¾	1?¿Þ~p¿éÿ¿bKr¿ÚdN¿0Ù¿wÃ¾®¾vÏ¨=Ð>ò>D ?ì??ç\X?Q2j?/v?%}?|â?s(?§{?Iùu?¤n?Âf?#·\?ÙÉR?QH?L<>?Ëô3?Ü×)?!ü?.r?­E?X~?Aø>B^è>;SÙ>
Ë>º½>ú ±>}K¥>o1>2Ê>ë>`áy>IÙh>2ðX>8J>8<>I/>D9#>ú>H>Ýº>xBõ=)Nä=Ô=¥ÎÅ=Ä¸=a\«=}=o=%=~='No=3µ^=fBO=}á@=3='= s=ã©=©=cú<J+é<üØ<FìÉ<
è»<ÌÜ®<K¹¢<tm<Sê<û!<ít<nc<»){¿*¹I?4:>ôu¿)TK¿nî¾Pý¾YU¿¥u¿îK½mú??Ã©¾²¿Ýåo?(÷¿`L}?¢¶¿XBw?Ü8¿}Xè=¬-?
E{¿ X>9æb?Î0¿Ý¸L¿xö?=Ao?¯-½Wt¿aF0¿¡å>Yâq?H X?¿§2>Å¿!x¿vi¿¹Àÿ¾£·o=n?d?òÿ?åg?=*?y¯>$Q×ºÃG¥¾s¿!K¿­®l¿I}¿ý ¿Èòt¿ßa¿n G¿j)¿Õ¿íÃÎ¾ÿg¾Êv¾.Õõ¼÷¥=é;>Êâ>tQ¹>`hà>Ù±?ÞE?}?ÜN+?6?c?? ÚG?Ã O?}U?ö[?²ä_?äd?BËg?Gþj?fÆm?;1p?¿Jr?ot?}²u?ðw?ÌBx?/Ky?l0z?(÷z?j£{?µ8|?º|?/*}?P}?wß}?](~?g~?7~?Í~?¤ö~?0?ù8?¤S?¼j?¼~???¬?L·?
Á?zÉ?ÉÐ?×?Ü?Wá?så?é?ì?Âî?ñ?ó?Íô?Nö?÷?»ø?´ù?ln¿î?mGÿ¾b¿øÆ½ÏÁ>ÀúÈ>¢L¼ò?¿÷b¿óªé>ãú3?Ö|¿(O.?õõ®¾HÓE>°¾q0?¦]¿`t}?k¿Í¾þ?¾ß$g¿ëÂë><©k?ÿÛ_¾¿v¿Ò¾m$?]m|?" ×>dç¾(Dw¿5^¿+í¢¾ÿ°>U?ãÿ?[?OEþ>B=¿©Ã¾@Ø7¿KÞl¿&Ø¿Zöt¿LS¿®Á!¿Ç Ð¾^r.¾ç»p=o÷>~Ëè>l?á<?­V?«h?'u?£|?	Å?la?$|?À«v?~o?Jg?ÛÀ]?6ßS?F«I?X??5?êê*?L!?ïs?ø<?j?ëú>^ê>4æÚ>@Ì> ¿>q²>z¦>¶Y>´ß>>Ç{>%j>Z>ç K>B©=>l¡0>az$>Ø%>C>Ð¾>Ñ&÷=Sæ=¼'Ö=ôUÇ=#¹=£¯¬=ò¸ ==97=L=(q=¸n`=YÝP=ù_B=ã4=iV(=÷§=ÚÈ=¹ª=wü<ãùê<ªÚ<è|Ë<ß\½<Â7°<-ü£<í<ñ<0&<9óu<àd<¼¾6Ë¼¼é]?¼/î¾¹Ì~¿Ról¿Ísk¿ô¿Âc)¿rî>ÞÈc?~6¿$p ¾N};?Øp¿h-{?ËÌv¿³dY?**¿ô
¾ÜþT?Jtj¿¶ü;ÜHu?¢Ü¾£>c¿¢ÿÇ>ÌÎy?cli=°Ni¿¯}D¿d{*>oNh?l]d?><þ¾T±r¿{7p¿ÿ¿
Kõºpe?3^?j¶?¸l?.&2?Í5Â>UÑ=¸¾Ù¿¹kF¿"ìi¿B|¿·¿Òv¿
d¿§J¿èÌ,¿Õd¿ôÕ¾r¾ ô%¾&¶-½fb=11>õ>eÞ´>jÜ>æÒÿ>+°?4°?¯*?Gü4?>?G?dN?ÙT?n|Z?g_?ã°c?clg?Ã«j?«~m?êòo?¡r?uîs?¸u?îv?$x?0y?\z?*ãz?{?±)|?­|?é}?}? ×}?!~?*a~?·~?ÕÈ~?ò~??á5?õP?ih?¹|?P??¾ª?+¶?À?¡È?Ð?zÖ?Ü?Üà?	å?¦è?Çë?}î?Öð?Þò?¡ô?'ö?y÷?ø?ù?=ÅO¿ó'?«¬>NØr¿êÇ?¿+¾õ\y¾#_¿u³|¿+¿ØQ?8¤>w¿Âh?¡'¿¢Y?2d¿xn@?{¸w¿&1l?é°¾~¿0{?ûñÊ½Àöv¿¬>ô5x?ÿ¹½`{¿ÿp¿Ì\?×?£7? ó»¾p¿¸g¿NÕÆ¾>4XK?Û?Q3b?n
?÷|Ä=e®¾È+0¿Yàh¿Y¿êXw¿µW¿Q'¿)¼Ü¾¶H¾;¶=H~>Ïaß>s?DÇ9?r·S?ææf?¶t?A
|??®?­|?æWw?aSp?bh?ãÆ^?ñT?¸ÃJ?¾q@?Â$6?ü+?V"?Åt?3?V?IÃû>®±ë>xÜ>ñÎ>°À>ÓÁ³>0Ã§>Ä>	õ>,>¬}>Ðdl>Ì?\>v,M>]?>·ù1>n»%>"Q>2­>ºÂ>ù=qÔç=åË×=;ÝÈ=|õº=á®=Éô¡=§»=òH==Øs=;(b=JxR=sÞC=G6=·¡)=NÜ=Ðç=Éµ=qþ<{Èì<YÜ<Í<´Ñ¾<¸±<?¥<fÆ<<f*<×w<Ê¢f<¶?¨ÅP¿¨ q?:¢>)¿»Ós¿Kx¿sçO¿Ý#¾X?C¢?xyr¿VÉ>Á'Õ>!|A¿6T\?YiU¿iÖ(?¢(¾>zÅ¾Èep?DN¿I¾o½~?¼Ô¾ús¿1­z>[=?ÅÃA>xxZ¿òV¿Ðt=Oh\?­ n?ÀÞ®>£Ù¾Pèk¿Íu¿¿*½¾ï>Ï>W?®Ñ~?Âp?»½9?m¥Ô>:=Jý¾
¿UA¿¯üf¿{¿¦×¿ þw¿Yf¿ÈM¿º!0¿½ê¿@Ý¾Iu¾§i3¾Üz`½½m=<&>ë>Tg°>sgØ>Ù=ü>g?ôF?­Ò(?Xã3?I=?¯*F?&§M?3T?çëY?é^?Cc?Ôg?¥Xj?j6m?#´o?Þq? ¿s?¥`u?ïÊv?6x?Çy?z?Ïz? {?|?õ|?}?´w}?yÎ}?¥~?ÃZ~?,~?Ä~?Zî~??Â2?AN?f?²z???m©?µ?¿?ÆÇ?OÏ?ÖÕ?}Û?aà?ä?Jè?wë?8î?ð?ªò?tô? ö?W÷?ø?ù?-nY=NØ¾~o?vÛ¾ê¿ëpX¿ËG¿½q¿	¡k¿Ýù!½d~?0V¾F¿5Ó?¬a¿ëUD?~¿H¿<!i?dô¿¢¬K?õè½tA¿~k?íÑ»=þ~¿|>Ôâ~?Á=ç³r¿ât¿å>µ?!?á¾\$h¿yºo¿Vé¾áX>áÒ@?
~?h?§?R¾>	¾ /(¿d¿Î~¿]ry¿òÏ[¿Ñ%-¿,Mé¾Ëa¾yo:<(h>àÕ>?ø6?_NQ?,e?ýðr?gk{?æk?5¹?D	}?¶ýw?Õ"q?ðh?6É_?Ç V?¢ÙK?[A?1:7?ß-?>#?¯t?P)?A?¨ý>0Zí>4
Þ>Ï>àêÁ>¼µ>þ¨>©>/
>>¸>J*n>^ç]>å·N>^@>îQ3>iü&>^|>Ä>Æ>Zïú=é=pÙ={dÊ=Ïa¼=V¯=0£=¬á=§Z=á=,Ýt=¼ác=9T=ì\E=«7=í*=¥=Å=ØÀ	=N1 =î<Þ<+Î<FÀ<­í²<ð¦<ßò<-1<.<Ð»y<weh<£?âs¿uÄ´>QJg?áÓ<kµ¿ä ¿´m¬¾C&È>½Ì?®#û=&ó}¿ý,"?0i=ºñ¾ÿG$?òÛ¿IÓ>½1<ë¿%X~?æ(¿ëÊÈ¾Òë~?ÅTµ½Ç7}¿¬Ø¾=hp?ÔÝ¢>FH¿)Öd¿p C½iNN?iÔu?Õ×> ³¾ÙËc¿71z¿²c(¿Ó¸ú½EðÕ>²³O?IR}?ést?½ A?<Áæ>ìÜ=6f¾28¿	<¿åàc¿cµy¿Âû¿Uy¿Hh¿D{P¿Jh3¿f¿ÿ,ä¾ïo¡¾$×@¾\½T>=þ]>µ!~>Zì«>¸`Ô>¤ø>~?ÁÛ?Ø'?ºÈ2?î¦<?àPE?xèL?ÜS?cZY?£j^?Ôb?¬f?íj?£íl?æto?0§q?rs?D7u?§v?æw?Æúx?µêy?¸ºz?o{?P|?¾|?}?Èm}?áÅ}?3~?PT~?~?2¿~?*ê~?a?/?K?¶c?§x?Ê??¨?á³?¾?êÆ?Î?1Õ?íÚ?åß?3ä?íç?'ë?òí?^ð?vò?Gô?Ùõ?5÷?cø?hù?t^?àrr¿`?¼µ>`Á6¿?°}¿Èé¿¶Ww¿Öü¿)!î>]9o?q@¿±ä¾uÿo?ô}¿°Ýo?S2p¿É}?Ku¿{ ?Í÷û=÷3a¿R?Öï>û~¿È¼l?3ùT>¸e¿÷7¿ØA¯>#í|?íh+?'A¾Ið]¿.v¿¿º>x}5?Ç·{?9n?ÔO ?|äD>Õ»¾ëå¿ÌÒ_¿­U}¿B{¿éÙ_¿ð2¿f±õ¾_ {¾ÎJ¼ÐR> GÌ>Æ?ãe3?¡ÖN?kcc?Âq?Áz?90?Ø?ap}?.x?ðìq?0Ùi?ÒÇ`?òW?íL?áB?éM8?À.? "$?©s?Y?[+?	Aÿ>âï>8ß>¼Ñ>¢OÃ>Ka¶>Å9ª>3Ñ>&>Õ>X»>ïo>È_>4CP>FüA>ª4>T=(>§>ñÚ>tÊ>Óü=Zë=Û=´ëË=Î½=N©°=ll¤=¯=[l=©=}·v=9e=&®U=cÛF=}9=Q8,=úD =¹%=æË
=×)=©eð<¶ß<Ë.Ð<]»Á<¢H´<ÑÄ§<W<ËH<Ñ2< {<%(j<ä_ý>y_¤¾¼Ã÷¾£To?9D3?"L	>µMÕ¼M>JKR?d b?sK¶¾)V¿0e?.²¾ë7¾ã²>±¾4h>ÿ¶>glI¿~?¿xó¾N¿UÒu?|a¶={õ¿©Zx½çfz?põá>7ÿ2¿	p¿*4¾Ï$>?g{?+.ÿ>r¾gZ¿)]}¿4¿`:¾Ît»>G?"9{??¥w?åëG?ø>*a>-D¾rhø¾ïb7¿]`¿³0x¿û¿Óz¿Ý½j¿CHS¿U 6¿
Ø¿ä4ë¾b¨¾Ù;N¾Þó¢½ç=X{>3t>m§>èUÐ>9õ>Áâ
? n?4O&?q¬1?­;?°uD?(L?åR?áÇX?Üê]?8eb?«Kf?°i?U¤l?35o?Ýoq?j_s?u?Ýv?³Æw?ßx?Óy?E¦z?O]{?óû{?o|?ü|?Êc}?9½}?³
~?ÑM~?õ~?Rº~?ñå~?¹?r,?ÊH?Va?v??ô?Ã¦?¸²?½?Æ?ÐÍ?Ô?]Ú?hß?Çã?ç?Öê?¬í?!ð?Aò?ô?±õ?÷?Eø?Nù?øËb?I¾R¿+%K>k?Õ8½T,¿ÕÚO¿&¿¯'6½©bX?sÃ(?Û[¿êC)½F¯;? ¾x¿kö?2Ó¿.|?ÿX¿¹?Ì>M´>dcv¿#Ó/?·é>Üív¿JÀ:¾Õz?jê°>ï¨T¿xÄK¿Rm>±9w?F=?l¾Å½jR¿´{¿^¿=3d)?Eqx?»s?:*?%u>ÐSV¾îS¿ÂÆZ¿Ò{¿Ç|¿ªc¿uð7¿;ó ¿¾ ½kí;>0Â>z}?H0?bPL?ëa?Kp?dz?|ê~?î?ÿÏ}?H6y?¯±r?Þ½j?±Âa?X?ÓýM?N²C?é_9?9)/?(%?´q??	?3 ?Ã¨ð>+á>ÔÒ>ö³Ä>°·>£t«>ø>î3>ñ>¶­>ª´q>
6a>bÎQ>mC>6>.~)>®Ò>Àñ>DÎ>ª·þ=í=(¸Ü=årÍ=c:¿=}ü±=7¨¥=¯-=~=n=Êx=´Tg=IW=ØYH=ss:=-=Ny!=­D=ôÖ=_"=>4ò<dá<l¿Ñ<00Ã<£µ<±©<ÐK<h`<7<f}<Òêk<úzí¾¤V?néz¿t^È>=½?UO=?±p?ÛB?*¿?ûË?@¿³$¿È?Y.¿ºr>y<°o½YÎ%¾kd	?y¿j¿Òo?Êú¾§:¿ÆÅc?Ë>é´{¿}!Z¾I;p?õ?÷¿®y¿~æ¾Ù,?ÌÍ~?¢N?E·H¾ÉO¿aM¿³ñ?¿	úv¾a >oî>?x?kTz?|N?}ð?G8>ë!¾é ê¾/2¿·&]¿Áv¿lÕ¿æ¦{¿Êl¿V¿É9¿ò>¿.ò¾hK¯¾+[¾óE¼½³¿<l>->j>ë¢>GÌ>Æeñ>ëD	?ÿ?Ä
%?0?¤±:?C?agK?)<R?c4X?6j]?õa?êe?²[i?Zl?õn?#8q?/s?ãt?n^v?§w?)Äx?[»y?«z?sK{?yì{?x|?òð|?¸Y}?´}?$~?EG~?I~?hµ~?°á~?
?A)?F?ñ^?t?6?f?j¥?±?¼?,Å?Í?âÓ?ÌÙ?êÞ?Zã?1ç?ê?eí?ãï?ò?ëó?õ?ñö?'ø?5ù?¤ú´=7d½M8¿jk?<Ù%?u&½0]¾ÓMÏ½örý>bÊ?ýDl>/¿!¾>?­Õ>h»R¿ÏÈr?Wv¿[½e?ÈÍ*¿´>~?±¿U0?úý?^g¿§¯¾ÎÌn?Æ·ó>§Ò?¿¶a]¿#îñ=¬n?ÑÄM?/o»¡D¿5B~¿ö®$¿%4;!?Bt?*=w?I4?`Ö>N¿(¾}¿|dU¿øy¿<~¿ø@g¿5=¿õ¿ ¾£t½«%>?Ô¸>`?kÊ,?Ð»I?º¨_?r?o?LJy?³~?hû?(~?Éy?qs?
k?Î¹b?ûY?O?ÃD?-p:?F50?.&?Ín?!?)þ	?a]?ÒNò>D»â>cÿÓ>ÚÆ>]ÿ¸>8¯¬>¸¡>H>æ >õ>ys>$Ýb>pYS>ÆÝD>Z7>ö¾*>Áý>>Ò	>ÞM >xàî=+\Þ=úÎ=¤¦À=§O³=ÿã¦=«S=¹=2=lz=,i=ùãX=LØI=h×;=æÎ.=¡­"= c=â=ç=Óô<	ã<PÓ<¥Ä<þ¶<Jª<Hx<x<;;<±h<­m<ÿ~¿eÔ?@I¿&É¾sC?ÜÉ?¶t?e¯~?o^?Â§%=ªy¿o6­½5´m?ò£h¿ÑZ?a¢¾³>Óáá¾8°>? é|¿+¡S?:ú]½'Z¿^nI?PÜ>Ïp¿\¸¾ë"a?[)?Ù ¿ 5~¿d*¹¾¸L?ÿ?|í#?xî½ÿC¿uÿ¿$J¿qE¾¤Ì>
À5??u?|?:®T?k?î\>sñþ½rÛ¾÷,¿Y¿Õ«t¿¿ |¿Ç¼n¿¿¦X¿Íã<¿¿lù¾+¶¾}èh¾Õ½þA<Sõ=JC`>ód>^4È>OÀí>¥?¥?Ä#?æn/?º´9?.»B?û¤J?&Q?èW?°è\?>a?Èe?/i?&l?n´n? q?Mþr?J¹t?»9v?Cw?¨x?j£y?ë|z?v9{?âÜ{?j|?=å|?O}?¶«}?û}?­@~?|~?u°~?fÝ~?S?
&?>C?\?or?g?Õ?¤?a°?»?JÄ?KÌ?9Ó?9Ù?lÞ?ìâ?Ñæ?1ê?í?¦ï?×ñ?½ó?aõ?Îö?	ø?ù?ÕYJ¿G?>Ï¿ñ´>pF~?c?C1°>Äö>Îa?Å»b?î¸¾ðÐn¿u8?5ÿ=Æí¿¶ÖI?ðS¿';?¾á¾ïË½Þº=?4v|¿öÓ­>õ_A?õO¿÷äû¾ôÒ]?Ã?j'¿Vk¿@è?;]c?ò[?¶Ý¶=_·5¿AÕ¿ç2¿\~½?/o?z?=?ª>4¤õ½Ue¿®O¿ Éw¿ßñ~¿)j¿
B¿áÜ¿£¾pÂª½!U>Yü®>ß5?g)?G?ô¶]?ëm?Ú}x?à@~?ÿÿ?·x~?VUz?+t?±yl?(­c?ÑZ?¾P?ÍÒE?²~;?ç?1?Z2'?ój?Üø?æ
?;?ôó>IJä>gxÕ>M{Ç>ÞMº>é­>¢F¢>î\>µ#>>B>u>d>^äT>_NF>ö±8>®ÿ+>Æ( ><>ÉÕ
>à?>]£ð=& à=1Ð=ßÂ=Í¢´=Â¨=¤y=e¡=ó=ZF|=¢Çj=ß~Z=½VK=[;==/0=ôá#==í=n=hÑõ<Áä<ªàÔ<×Æ<Y¸<r«<¿¤<£<p?<~¦<,po<öÏ¿qP ?eø½{o¿Ví=
®J?K]p?äM`?;;ñ>Ó¸í¾Çw¿òk¸>ú1?tÐ¿uS?x¿L?¸.¿$Æe?½~¿)Þ+?y©)>`Ëp¿Á'?4L?à^¿ÿ¾mM?!A?ëSÈ¾¸ÿ¿mKë¾ÿ?:ú~?×Q4?äê½7¿r¿DT¿~¶¾]Q>C,?²aq?ã&~?Z?×®?Ñ¤>[Á¹½ÄáÌ¾~ì&¿ÂÂU¿<¬r¿Í¿Ñz}¿¯p¿»7[¿ºî?¿ì ¿7õÿ¾½¾6/v¾ÙÒî½¨9vß=©BV>SÛ>ËÄ>äê>X?Ô?|"?©M.?O¶8?àÛA?ZáI?
çP?s
W?Kf\?a?Ó$e?°h?EÅk?[sn?|Çp?8Ír?°t?Äv?1gw?½x?My?hz?W'{?-Í{?æ\|?qÙ|?\E}?Û¢}?Üó}?:~?Òv~?y«~?Ù~? ?Í"?o@?Z?Tp??A?±¢?2¯?º?gÃ?Ë?Ò?¥Ø?ìÝ?}â?ræ?Þé?Öì?gï?¡ñ?ó?9õ?«ö?ë÷? ù?¡Hq¿xËw?ÖB-¿:Ü¾èIN?z?	W?úd?lW?Äý?Ó,¿sW.¿é¼p?X¤±¾p¾ùÎ	?CÆ¿æb ?ê:¾l±¾®Ë`?/m¿ò>¬·]?¨D2¿·ð ¿"G?4?j\¿g-v¿æ½oU?¦ìg?zÏ9>p%¿:¿¿Ë@¿1T
¾?=i?!}?KIF?:ÖÁ>¯H½Ç!ø¾Ù¥I¿(Fu¿&¿L»m¿ÙëF¿¶©¿Õ[¯¾Û½Ùñ=¥>¬þÿ>÷%?phD?º·[?l?¦w?Ý}?Ùû?ËÁ~?AÛz?ßt?ÍPm?¸d?[?Ò Q?ÚßF?w<?I2?w5(?$f?Ñê?sÎ?4?tõ>¢Øå>ÞðÖ>OÞÈ>»>#¯>Pm£>&q>\&>>Âw>Ü*f>*oV>Ý¾G>Ã	:>T@->¾S!>ê5>}Ù>Ú1>5fò=¤á=KÒ=Ã=íõµ=[©==³=³= ~=l=Ã\=-ÕL=M>=ve1=E%=¡=ø=õ=û÷<pæ<IqÖ<©Ç<s´¹<RÐ¬<7Ñ <@§<¥C<£<Ù2q<º«>¾_u<?¼g¿B¿ü@[><+
?óä>ùÌ½(CX¿hÛ<¿u;?
$®>ùp¿ÊÝx?¾W¿XaJ? z]¿¼{?p¿ùô>KÁ>=u}¿¹òÿ>³º7?]G¿/M ¿5?2V?lw¾Za~¿ÅZ¿ÄØ>À{?>\C?16=1)¿H§}¿Ô]¿@Ó¾Ïí>âë!?Üñl?.H?í_?$·?±¦>¶´h½£ñ½¾$!¿ÞÒQ¿Mp¿ß~¿o5~¿Tr¿9´]¿êB¿Û1$¿¼`¿ÎÃ¾]µ¾×¾Ýe8¼É=<L>JN>tÀ>iæ>§_?(§?Ó2!?Ë*-?e¶7?7û@?I?Õ:P?tV?ã[?/ `?/Ád?^Yh?ßyk?Ó1n?p?Ér?Èct?ïu?åFw?ºpx?sy?ôRz?{?[½{?0O|?Í|?;}?ð}?"ì}?W3~?q~?t¦~?»Ô~?Ñü~??=?§W?6n?¾?ª?P¡?®?þ¸?Â?ÀÊ?ãÑ?Ø?kÝ?â?æ?é?ì?)ï?kñ?_ó?õ?ö?Ì÷?æø?i¾îÀ>Í]ú=úr¿>>Ôb?.2?ix}?Ï<N?U)=q¿Y¾wI?}ã-¿E>Wh>¸ ¬¾àj>±Eº=jp¿|Gw?ì&R¿½ùüq?èô¿F?¿ÿ¥,?L?ÄLÝ¾;ú|¿{g¾;E?
q?=J>nì¿F ~¿AL¿Ó¼P¾ÄÑä>¡pb?±à~?ý{N?À¬Ø>frò¼òå¾NC¿ønr¿àî¿p¿K¿dZ¿»¾©¾¿êÄ=õ>«y÷>y"? ªA?(«Y?ºk?Ãv?-o}?öî?X?ÁZ{?²u?Z#n?{e?\?J'R?ÁêG?y=?×P3?g7)?_`?ýÛ?¤µ?×ô?<÷>Lfç>ÉhØ>ß@Ê>Ìé¼>=]°>Á¤>->Ý(>ôu>Çx>{Ñg>ÔùW>@/I>za;>è.>§~">L>)Ý>Í#>)ô=ýGã=^Ó=BëÄ=I·==ª=Å=´Ä=p=Ýú=:n=¥´]=SN==@=½°2=J&=uÀ=%={=nù<~è<çØ<{É<f»<2®<®ý¡<Ý¾<ÙG<È<õr<Ay?4!m¿q~?X6¡¾{¿ë[í¾*/¢½0¾ª¿ùÇ¿5¬¾ìÝt?¿½.á;¿Fã}?ÖVy¿~q?02y¿]ð~?ßúQ¿«>,1?x¿Æ§>½]T?>Ú)¿=¿Ü?Úf?Çã¾_y¿Zn#¿`©>,Xv?ØïP?cnÿ=R¿óz¿e¿£Öî¾ú¿=T?6òg?Kã?}õd?G%?v¤>ÖC»¼ãÐ®¾Ô7¿²ºM¿b4n¿MÏ}¿[Ð~¿dùs¿þ`¿¹ÕE¿l'¿ã¾¿åÊ¾7M¾¾ÒÈº¼÷¨³=51B>ò½>nå»>r¸â>º?¥0?Xç?N,?þ´6?2@?lVH?O?ÜU?ç^[?-`?ß\d?h?ó-k?Õïm?;Up?jr?8t?	Êu?^&w?Tx?Zy?¾=z?´{?l­{?aA|?Á|?²0}?õ}?Zä}?,~?.k~?e¡~?ZÐ~?ù~?@?Ã:?0U?l?å??í?Í¬?ó·?Á?øÉ?6Ñ?{×?éÜ?á?°å?7é?Dì?éî?4ñ?0ó?çô?dö?®÷?Ëø?O22?
¿ÐU?jb¿±¿¯Ã>pþ>?	/<?};»>pPí¾fW}¿E >a?Dh¿Î£ò>ÛGã½¥0Ú¼¦Z½e¸¶>U¤B¿ ì?p-¿f¾ïr}?ÑEÎ¾(æX¿Õ?Äz`?Áò¾ã¿Òo¬¾í`2?|Õx?-x¸>vL¿Kz¿e>W¿¬¾cÆ>ÐZ?Ò?÷V?Ãÿî>a½<ÞÑ¾f©<¿Eo¿óû¿FBs¿%P¿Òí¿£Ç¾²¾_ã=>Ýî>Ûí?ýÝ>?`W?£i?ÍÔu?U÷|?WÙ?Z=?ÑÓ{?a8v?Vñn?mpf?m]?#+S?óH?µ>?#W4?'8*?£Y ?_Ì?1?õÐ?»Þø>Góè>&àÙ>ý¢Ë>87¾>©±>ö¹¥>>5+>´g>(z>ðwi>^Y>J>¹<>kÁ/>©#>"c>Ëà>¹>Ãëõ=Ûëä=hÕ=jWÆ=¸=õÒ«=|ë=XÖ=+=ê=òóo=O_=ÒO=,gA=ü3=æ~'=eß=/=ý==û<ùÌé<Ù<MxÊ<Yj¼<V¯<%*£<yÖ<L<í|<2¸t<Ì7?G[¿ûÇ?¸ïî>4X¿Þ¯l¿Mv*¿8-¿ÏDn¿×b¿³1>SØ|?ýÒó¾«2Ö¾dma?/k~¿Àè?&}¿$o?ÓH&¿.¼É<~¥<?¿v¿3>Nj?û¿ÈCU¿&!ö>s?xó¼\q¿A7¿¯p>În?ò\?#Q>
¿M`v¿ôk¿hà¿N=R?Çeb?è÷?Bi?-
-?2¶>426<c¾:)¿	{I¿Þ¼k¿2ò|¿K¿Ôu¿Ïnb¿Q±H¿¢*¿Û
¿HÑ¾ÜÞ¾.¾*¬½º=å 8>e*>ÌÃ·>ß>?O¸?#?6à*?²5?Ö5??!G?"ßN?5DU?èÙZ?¹_?á÷c?-ªg?áj?c­m?p?à7r?t?F¤u?w?8x?éAy?b(z?/ðz?_{?y3|?µ|?A&}?è}?Ü}?Ï%~?Me~?L~?ðË~?3õ~?ñ?å7?´R?ìi?~?s??«?ç¶?³À?/É?Ð?äÖ?gÜ?,á?Nå?âè?ûë?ªî?ýð? ó?¾ô?@ö?÷?°ø?üðz?/Ò~¿Äu?6¾&w¿³®¾(*D>ù`>WY?¾#X¿¬N¿¸?Ñ|?Í¿µÒA?ö,ß¾Ø4>'³¾À ?öEf¿Á=z?"ÿ¾0*ê¾½®?²@p¾÷Ul¿ZsÖ>ðÀo?7¾gÜ~¿ßÕâ¾Ë§?r}?vä>ÝhÛ¾íu¿a¿¤­¾¬§>îcR?lô?ß)]?Û`?¾Ôy=k¤½¾Rº5¿hÉk¿Z½¿²¨u¿ZT¿ðb#¿ÑÓ¾å¤5¾U=ê>+æ>×U?<?jU?ÿh?XÛt?u|?ûº?Ño?oF|?Üv?»ºo?Tg?
^?Z,T?úI?(§??ú[5?µ7+?íQ!?÷»??¬?ú>ê>õVÛ>¦Í>G¿>ÉÏ²>îß¦>¨¬>f->TY>O|>;k>Å[>´L>¦>>Ü1>OÔ$>­y>dä>>x®÷=®æ=kÖ=ÃÇ=0ï¹=¨­=h¡=ùç=ä=©×=]­q=bê`=rPQ=ËB=FG5=4³(=Tþ=9=õ=°ý<t{ë<"#Û<íË<LÅ½<ð°<V¤<î<BP<o<Þzv<¤J¾GSÄ½ÒN¾v?·¾	t¿>B{¿Ê¾y¿¢}{¿/	¿Q?ÈQ?â^K¿. ½û<'?fff¿&Wu?Ño¿M?¹²ß¾ùW¾ó]?2¨c¿ÊéK½­Úx?ÀÉÄ¾[vh¿}³>-Ú{?)ûÀ=6e¿Ú±I¿×>ö3e?rMg?×>H$ô¾¤íp¿æq¿QÙ¿­ù¼Ñí ?ðO\?û?µÊm?ÞN4?¼gÇ>­8=¾ù¿²E¿*i¿®ð{¿Ú¦¿Ãóv¿v¬d¿©|K¿1½-¿rb¿,õ×¾j¾Þ¹)¾ï;½Æ=Þ.>¼>£³>ìJÛ>xÒþ>+>?8K?¸)?Ã­4?"Q>? ÆF?§/N?ÙªT?TZ?YD_?8c?±Qg?j?|jm?aáo?fr?<ás??~u?äv?}x?)y?Þz?Ýz?5{?w%|?h©|?¼}?Ì~}?Ô}?ø~?`_~?+~?~Ç~?Zñ~??5?4P?Âg?(|?Ó??_ª?Ùµ?É¿?dÈ?ØÏ?LÖ?ãÛ?ºà?ëä?è?±ë?jî?Åð?Ðò?ô?ö?p÷?ø?ñ¹>ç+¿ØNÑ> ?"«a¿ßX¿:Àá¾ <¿¾Ä.¿Å¿èß¾¸Á_?Éj>i2p¿õÃp?Ë 6¿wÆ?«H¿J?){¿Çf?¤È¾a!¿x?¸q½ÁPy¿¾>ßz?@½Sêy¿4¿?½Ò?ÆÔ?Ü²¾øn¿Üi¿ìBÎ¾Ó>¶2I?G?Åc?³ò?¾cÙ=¦l©¾.¿Oýg¿*3¿<Ðw¿ xX¿°¸(¿teß¾}M¾¿ö<~y>cÝ>?±?9?¯6S?Éf?¾Ös?¿é{?å?»?²|?[{w?p?Ð4h?ý^?ì*U?yþJ?Ò¬@?X_6?6,?<I"?Âª?\g??!ü>,ì>3ÍÜ>ÜeÎ>øÐÀ>´>¨¨>À>o/>ÔJ>¾~>\Äl>
\>ÆM>h?><B2>ÿ%>,>óç>zù> qù=x3è=f$Ø=§/É=<B»=WJ®=Q7¢=ù==ÄÄ=Äfs==b=ÛÎR=/D=6=ç)=C=C$=î=@Úþ<î)í<¾³Ü<ïaÍ<> ¿<ÏÛ±<¥<²<vT<7a<=x<[n¿^²=?¡i¿gW]?Ð½ñ>sK	¿Åe¿ìxm¿B;¿s-½ä8f?¿ø>m2y¿±>ý­>»4¿ÿpR?¶K¿?+ÂH¾Þ¾ÿ}t?ÌÿF¿NMt¾ö?¾yh¾(v¿¼WZ>í·?Fî]>sèV¿ÂpY¿d3=Y?ìo?p·>½Ñ¾ÀNj¿äÓv¿ëI¿û½[^ê>`´U?À~?ìq?}L;?4}Ø>üË¡=Úq¾3ª¿@¿¶Xf¿åÊz¿Râ¿
Ix¿¼Ôf¿7N¿Ó0¿t§¿vÞ¾Xî¾|>6¾¨+k½c=bò#>ú>v¯>©×> |û>>Â?ú?>(?ó§3?k=?êüE?M?T?UÍY?àÎ^?â+c?øf?Gj?!'m?Ú¦o?Òq?µs?ôWu?dÃv?¬þw?y?4ýy?ÁÊz?î|{?[|?.|?%}?u}?«Ì}?~?iY~?ÿ~?Ã~?yí~???2?¯M?e?Dz?0?µ?%©?É´?Ý¾?Ç?'Ï?³Õ?^Û?Gà?ä?6è?fë?)î?ð?ò?jô?øõ?P÷?zø?¼y¿ÀE>»OÜ¾CÀx?X¥¾ÓÇ}¿×d¿ÆU¿|w¿bòb¿µ«=æ?Â	B¾<¿Ùô?<g¿¨ýK?é£O¿ªRm?²¿IöE?H¤½9=F¿^{h?¿Èò=¾{¿±ý=5X?oÓ°=K q¿ñ"¿Þ>­~?"x??h¾Îf¿*¼p¿,oî¾öO>vE??3Ì}?&li?þ.?mµ>Ãç¾'¿âc¿]~¿V¸y¿he\¿î-¿õë¾­7e¾<e>gÔ>` ?y*6?
öP?uìd?
Çr?T{?d?¾?H}?x?³?q?9i?fí_?Ö&V?­ L?­°A?<a7?63-??#?À?÷K?'b?ÃÁý>í>áBÞ>ÆÏ>IÂ>%Aµ>%+©>\Ó>O1>3<>:Ö>Rjn>-#^>»ïN>y¿@>3>½)'> ¦>zë>Oë>½3û=8×é=Y«Ù=¼Ê=C¼=¯=6]£=3=O=Ý±=) u= d=BMT=ðE=ËÝ7=Ï+=0<=L/=æ	=hT =gØî<ZDÞ<¿ÖÎ<0{À<­³<¯¦<N<«X<[S<6 z<FO¿T.{?g¿Ëôq>bAr? >Bå¾u¿ù´¾èì>WÙ?ñ¼]=¨\{¿­-?¼æ#Ú¾I¬?ù¼¿=ú¿>!bk=SQ"¿Ï,?¨ú!¿¡]Ö¾ß1~?ð½?~¿:u=D?ú«>ØrE¿{­f¿ò½)%L?T¿v?ó~Ý>((®¾áb¿ç¶z¿&*¿¦¾Ñ<Ò>N?¸}?-ït?H B?ÈHé>^ç=ka¾<¿f×;¿÷lc¿y¿ãý¿y¿mçh¿·áP¿Ý3¿®ã¿,å¾k¢¾ì»B¾0½;¨7=µÔ>ÿº|>J«>ÑÎÓ>)"ø>D?M¨?cd'?® 2?»<?2E?pÍL?:uS?ÁEY?¥X^?àÄb?ñf?ùi?Qãl?îko?fq?¯s?e1u?ñ¡v?¨áw?òöx?cçy?×·z?l{?&	|?ß|?z}?`l}?©Ä}?%~?fS~?Ë~?¾~?é~?Ý?+/?&K?ac?]x??G?é§?·³?ð½?ËÆ?uÎ?Õ?ÙÚ?Ôß?$ä?àç?ë?èí?Uð?oò?@ô?Óõ?0÷?_ø?¿zøf?#ew¿æW?WÑ>Á×,¿¦¦{¿ò¿?t¿@a	¿çüþ>;äk?%¿¸Ö¾dm?¶~¿úq?(%r¿?p~?ls¿FJ?D>c¿läO?E«>È«~¿sô¼|W?D_>é d¿d8¿-C«>|?LÅ,?Þm:¾$]¿v¿'¹¿M~>u¥4?a{?ín?!?©gH>¾ÄJ¿òx_¿¸<}¿`{¿ö!`¿3¿bö¾HÑ|¾Éùe¼9P>ÛË>C?'*3?µ¨N?Bc?I¬q?p´z?+?ãÙ?~w}?c¨x??ûq?Âéi?õÙ`? W?® M?º²B?¥a8?&/.?å4$?ï?ê/?(<?aÿ>F ï>ý·ß>æ&Ñ><iÃ>`y¶>cPª>kæ>3>q->ÀÌ>p>.­_>_P>ÂB>ÄÂ4>^T(>½>öî>Ý>Möü=îzë=D2Û=ËÌ=Dè½=©Á°=¤=Ë==ô=Ùv=ìºe=§ËU=ÙöF=)9=P,=[ =T:=ß
=¯;=àð<öÔß<KÐ<"ÖÁ<a´< Ü§<ê4<Þ\<E<áÂ{<Åk=ÄÍÜ>¡¾H	¿å¬i?×<?×ò;> ¨<[>X?pþ]?¯èÆ¾x|Q¿fh?Æ¡¿¾ Í½­!§><'¦¾ÛÛá= S>ÔGL¿g{}?	Nì¾Zc¿aÑt?5iÐ=çâ¿¢Å½D×y?òXæ>ï[1¿ì@q¿T&¾|è<?º¸{?¹÷ ?¾´®Y¿Y}¿@d5¿Tà>¾î¹>rüF?®{?ñÚw?gH?ºÄù>9	>¬¹A¾d÷¾67¿j[`¿3x¿ù¿£z¿Zäj¿ýzS¿ïÚ6¿í¿µë¾pá¨¾®1O¾Å¤½ý¬=³>B|s>Ê§>sÐ>Äô>Å
?UT?ø7&?÷1?;?åeD?·L?ùØR?R½X?§á]?2]b?¯Df?ªi?l?0o?àkq?ó[s?
u?Bv?pÄw?Ýx?kÑy?Ì¤z?\{?Øú{?y|?¼û|?c}?¼}?(
~?YM~?~?ø¹~?£å~?u?8,?H?*a?rv?á?×?ª¦?£²?½?üÅ?ÂÍ?~Ô?SÚ?_ß?¿ã?ç?Ðê?§í?ð?>ò?ô?¯õ?÷?Cø?/gû¾¥b?ökR¿áI>Æk?½1½<,¿ËO¿À%¿>±0½X?t(?ªÿ[¿+%½»;?,±x¿M÷?ÿÔ¿À|?ëW¿÷Ë>VÝ´>³lv¿é»/?éëé>æv¿%;¾¹z?A±>aT¿'ÐK¿m>Y5w?»=?$KÅ½·
R¿){¿Ö¿ãÃ= \)?ox?y s?L*?°u><8V¾£N¿ÃZ¿Ñ{¿ZÈ|¿Ô¬c¿¥ó7¿éö ¿?#¾ýÙ½ìß;>GÂ>þz?F0?ÕNL?Éa?p?ò
z?Oê~?î?7Ð}?¤6y?%²r?h¾j?HÃa?¥X?xþM?ô²C?`9?Ü)/?<)%?Nr?4?	?» ?Ä©ð>,á>ºÒ>Î´Ä>M±·>bu«>Gù>4>>I®>¾µq>7a>SÏQ>ómC>í6>ñ~)>cÓ>iò>âÎ>Ð¸þ=í='¹Ü=ÓsÍ=A;¿=Ký±=÷¨¥=a.=²~=	=êx=ÀUg=
JW=ÁZH=Kt:=e-=	z!=\E=×=ö"=X5ò<eá<_ÀÑ<1Ã<j¤µ<v©<L<a<¤7<}<_?vî¾»Í?{¿wÇ>gÂ?Z=?­Â?wC?ÿÂ?#?±@¿"è¿²Ê?F.¿ìr>ð{<]½&¾[	?GÏj¿Ì~o?³¸¾':¿3¸c?)K>5°{¿Z¾3p?1?:è¿×y¿È¾T	,?1Ï~?°Y?H¾EÂO¿.N¿bø?¿w¾Q >é>?¯x?äUz?îN?µõ?i]8>6Ö!¾)ê¾å	2¿$]¿¯v¿JÕ¿§{¿QËl¿#V¿}Ë9¿ A¿º2ò¾O¯¾F[¾TU¼½:[¿<Ë>8j>Qè¢>¡DÌ>cñ>ïC	?µþ?ÿ	%?Ð0?±:?C?ëfK?Â;R?	4X?èi]?Ùôa?Õée?~[i?SZl?äôn?8q?ë.s?}ãt?X^v?§w?Äx?L»y?z?hK{?pì{?ýw|?ëð|?²Y}?z´}? ~?AG~?F~?eµ~?­á~???)?F?ð^?t?5?e?i¥?±?¼?+Å?Í?âÓ?ËÙ?êÞ?Yã?0ç?ê?eí?ãï?ò?ëó?õ?ðö?'ø?K|ï>3¡>Æ,Ê½F¿)p?NÖ?îÚ²½©«¾åÕ¾ì>E?V>¤N~¿Ä°>,Øá>M:V¿=t?.w¿bìg?¼F.¿ÛÝ/>d©?úc¿-	?ëe?kbh¿#­©¾g½o?Jaï>ÁTA¿èM\¿ÃÕ >HUo?3¾L?J "¼XE¿~¿!²#¿ÌC<u?t?üv?¥3?*<>ÄÐ+¾Þ¿ÃU¿ñz¿zï}¿:g¿òÁ<¿±¿÷É¾#n½W*'>|¹>§?-?èI?É_?ÓUo?Wy?] ~?Éú?p"~?^¿y?bds?&k?[©b?
Y?	úN?Z±D?ù]:?X#0?&?Ü]?Óõ?î	?N?2ò>} â>æÓ>  Æ>ìè¸>"¬>ð¡>û5>>·>3[s>ÆÀb>ô>S>ÅD>C7>t©*>´é>Óõ> À	>¤= ><Âî=@Þ=ÔßÎ=8À=é8³=ÒÎ¦=õ?=`}=y=FLz=ðh=kÈX=§¾I=¿;=¯¸.=õ"=bP=Ð==
=Ïãó<+öâ<.5Ó<Ä<Gç¶<ë4ª<"d<Fe<È)<8H<ßDb?ßè|¿â¿~?[R¿m¯¾XK??ï4q?&m}?c?ë=w¿!ë½ÅKp?º¿e¿Ñ&?üE¾$¨>°ÅØ¾";?;2|¿§äU?<<½Û$X¿bpK?ºÉÖ>q¿^³¾Lb?Î'?Id¿;÷}¿¢¸µ¾­?Ìü?}È"?êtù½ðÒD¿ý¿ÙI¿]J¾¹©>³a6?|u?å^|?ÝFT?ªÛ?{Z>SÉ¾bÜ¾gï,¿öÈY¿¯Ìt¿%¿Ø|¿%n¿÷yX¿¯<¿¶a¿$£ø¾Çµµ¾5h¾µÞÓ½¤­N<(Êö=Éî`>·²>hzÈ>ÿí>Á?q§?{Ú#?</?»Å9?ÊB?²J?Q?å©W?fñ\?Õa?ee?ïi?&l?Æ¸n?Éq?s?#¼t?3<v?gw?hªx?¥y?P~z?¬:{?îÝ{?kk|?æ|?BP}?M¬}?
ü}?A~?õ|~?Ë°~?°Ý~??B&?nC?±\?r??ð?&¤?u°?»?YÄ?XÌ?DÓ?CÙ?tÞ?óâ?Øæ?7ê?"í?ªï?Úñ?Àó?dõ?Ðö?ø??."3¿¬1?]Ü|¿kè>Ûéy?eg?=ó>ÇøÐ>_äW?,£j?­?¾t¿w-?q»=C¿'ìP?ïY¿xøA?Áò¾r½®8?à}¿aA»>.ï<?åS¿|Ãñ¾ìu`?·?ô+¿kÜi¿W<	e?#Z?ìñ=±Õ7¿C·¿p1¿f½@û?.îo?¨+z?ÆO<?µS§>¾²¦¿zP¿èx¿Õ~¿k*j¿lA¿¿ [¡¾C'¤½¶b>hU°> È?±Þ)?vG?°û]?9n?lx?ºM~?ãÿ?(n~?Bz?ôt?û[l?*c?­ûY?]óO?ê­E?áY;?1?å'?H?Ç×?îÆ
?Ó?ºó>Þä>úDÕ>ÐJÇ>< º>£¾­>e¢>77>e >
q>| u>^Jd>y®T>F>8>éÓ+>øÿ>3ù>V²
>Ù>Óeð=ÔÆß=ÏKÐ=)áÁ=t´=ªô§=Q=|=.f= |=bj=ÊFZ="K=Æ
==øì/=ß·#=i[=È=ñ=Eõ<Åä<ý©Ô<ôæÅ<%*¸<aa«<½{<yi<í<q<íò«=¾â6¿ÇM8?êÞ¾d¿¹ö]>¢X?ú7w?Ù´i?ó	?¤ÁÌ¾:{¿>ÏD<?ïì~¿ K?¤ñ¿"P?-'¿`ea?w¿×ê1?Ö,>¦Cn¿ú½,?þ4?©a¿Ò2ö¾É_P?,r>?fGÐ¾ô¿«ä¾]ü?H??Ò(2?eí@½Wî8¿¿3ýR¿²¾HY>²k-?iñq?õ}?ºY??6X|>o6Ã½µçÎ¾¶³'¿'IV¿rôr¿%-¿å^}¿«Vp¿DßZ¿T?¿ßx ¿|ÿ¾¯¼¾þ_t¾Û`ë½Áõô:irâ=x W>z>Û¬Ä>ê>z<?N?n©"?;u.?Ù8?oúA? üI?uþP?èW?$x\?'"a?^2e?Ü»h?Ïk?C|n?9Ïp?ðÓr?t?Óv?kw?x?y?ßjz?Ò){?TÏ{?Ã^|?Û|?ÂF}?¤}?éô}?ñ:~?w~?(¬~?¬Ù~???#?Ò@?oZ?p?Ô?x?á¢?[¯?*º?Ã?¡Ë?¦Ò?ºØ?ýÝ?â?æ?êé?àì?pï?¨ñ?ó??õ?°ö?ï÷?Oê?c­}¿?»ðK¿qÛ¾8&c?£o? +C?< U?mÀ?Zt?¿³?¿­Gh?ªM¾Ã¾?])¿?År¾ ¾Z?f3q¿Ú¬:>÷X?îß8¿¿vL?/?/¿vMt¿äµ½ÝX?È¤e?c{&>å(¿æ¿(|=¿ßx÷½~ù?j?[«|?³D?N½>OU¬½çü¾gêJ¿Ñu¿z¿³m¿ôðE¿<z¿Õ¬¾(Ñ½û=Ñ§>iÞ ?n­&?_÷D?"\?ÉÓl?qÓw?hò}?ký?]³~?4Àz?Öºt?á$m?³kd?êZ?sêP? ¨F?DT<?2?5 (?2?¹?Ä?µê?õAõ>ªå>e£Ö>?È>=W»>ãâ®>§0£>I8>ñ>BR>¥v>ÒÓe>áV>ÿrG>úÂ9>Oþ,>/!>ü>¤> >a	ò=Má=Ã·Ñ=4Ã=°µ=~©=c=µz=>S=ö¾}=/&l=(Å[=nL=V>=@!1=ÉÖ$=nf=Á=ÈØ=»@÷<^æ<ËÖ<åAÇ<m¹<Ö¬<X <­m<<Çf<ý
K¿ 	>êÒ~½T½?v¿ö ì¾´>®ª%?[? _4=FH¿5M¿Qf*?N=×>½{v¿¥§s?aM¿À??CLU¿Ô¶x?kpt¿P?M«>Ïµ{¿Ô?|1?esL¿ä¿[À:?(R?  ¾òý~¿ý¿²Eâ>|?$a@?LIä<K#,¿É~¿\[¿-;Í¾}¤$>o$?çm??×^?¶?]õ>?¨½*	Á¾ÔW"¿¸¥R¿;ùp¿T©~¿~¿¹úq¿Ú2]¿5NB¿I#¿6®¿iÂ¾Y¾|m¾q¼ÝÎ=_MN>u>>	ÜÀ>¹+ç>:¶?ô?Ýv!?Ñf-?6ë7?)A?!EI?a^P?V?!þ[?Î·`?ÂÕd?Ekh?pk?Z?n?Pp?þ¥r?¤lt?8÷u?Mw?vx?xy?MWz?Û{? À{?R|?Ð|?1=}?È}?»í}?¹4~?8r~?}§~?¡Õ~?ý~?6 ?1>?)X?¦n? ?þ?¡?@®?4¹?±Â?éÊ?Ò?0Ø?Ý?%â?%æ?é?ì?6ï?vñ?ió?õ?ö?Ó÷?	:­¾-å÷¾<i?Éì½©H[¿ê_Ä>>Qu?²»~?`ê?Ð(c?(Í<>rc¿/"Ñ¾²Ç?5¿IZå<¢>WhÕ¾î5>r<[¿nr?1¤Z¿þ»Bm?â-¿ß7¿\4?`F?hî¾l}{¿°§G¾ãÊI?¼)o?i>}>ßÜ¿¥~¿ÀI¿5p=¾Êøì>>ed?÷y~?WHL?oÒ>ùK,½Uê¾ûE¿}?s¿ýÝ¿kØo¿åOJ¿/Ì¿6¸¾±þ½LÑ=Ö>Ôù>{p#?ÅlB?ê<Z?k?²w?l}?bó?ò~?J8{?_u?Öém?òGe?ÖÕ[?HßQ?{¡G?"M=?^3?ð(??ª?v?$¸?Èö>ßøæ>VØ>KßÉ>î¼>ã°>µB¤>19>´á>^3>Jx>"]g>-W>ÔÉH>Ù;>¥(.>[,">Õÿ>ª>0á>ä¬ó=aÔâ=±#Ó=ýÄ=ªë¶=O@ª=t=\y=L@=Ix=úÀm=C]=OêM=<¡?=U2=²õ%=sq=¹=À=0ïø<÷§ç<×<ÔÈ<ß¯º<Kº­<óª¡<àq<4 <H<çp¿Hþ_?
ñL¿H~?É¿økl¿w\¾}Ë=7lÒ<ýì¾o{¿xúê¾«i?÷À)=&M¿Oæ?þÓr¿Â³h?"¢s¿ò÷?Ìº[¿|¤>P)?þ¿IÀ>jM?wf2¿+5¿Ð!?m¢b?º]?¾={¿¾¿j¶>>x?ÑYM?9IÒ=½¿Æ{¿îb¿<Eç¾ñß=I?º`i?þÅ?)c?b#?^>9ñ½Õÿ²¾ÈÜ¿EßN¿TÛn¿Å~¿Þ¨~¿(s¿t_¿w	E¿Æ&¿OÒ¿µÈ¾}¾&¾Q¾ ¼²¹=³õD>öÿ>½>½ã>S.?ò?ÉB ? W,?ü6?W@?H?Z½O?bV?^[?ËL`?xd?*h?çBk?n?ep?¿wr?Dt?cÔu?X/w?N\x?May?Cz?Ç{?Ò±{?1E|?çÄ|?3}?p}?æ}?u.~?Ël~?Ê¢~?Ñ~?ú~?)?;?ÞU?ªl?h??O ?"­?=¸?ÛÁ?/Ê?eÑ?¤×?Ý?¼á?Ëå?Né?Yì?ûî?Cñ?=ó?óô?nö?¶÷?y¿.Ó>SÙb¾Wò"?z¿ñ[¾"?­`?I\?Ä	?ý¾®í¿¶ºr:ep?BóW¿Y¸>LÃ;~
¾K¤=hÑ>M3¿}~?ðä:¿tLJ¾z?ãê¾ÕÊP¿õº?È Z?+*´¾ÍS¿£¾Á9?àv?Ê©>*Ö¿Cö{¿ÖS¿W~¾!!Ñ>]?8?ýS? ]ç>ú.Â8áJØ¾Hü>¿ep¿ýÿ¿ù_r¿ N¿¿£|Ã¾k¾
l§=£>Øñ>( ?^Ö??ELX?&j?6(v?Ç!}?Èá?5*?Îª{?~þu?Öªn?ä f?Ð¾\?ÙÑR?xH?wD>?ßü3?Äß)?Ï ?y?ÇL? ?wNø>~jè>Í^Ù>ó(Ë>OÄ½>£*±>T¥>ï9>(Ò>_>Oïy>Næh>\üX> J>¤B<>íR/>{B#>>I>RÂ>]Põ=[ä=Ô=ÞÙÅ=7'¸=f«=&=x=Y-=Í=Â[o=ÝÁ^=/NO=uì@=Ì3='=x|=²=R§=¥ú<8é<fÙ<Ä÷É<»ò»<Àæ®<Â¢<v<Xò<q)<Ye¾5i?¢£y¿ÕqE?+(S>?t¿ N¿f÷¾â¿°ÄW¿9st¿S°Á¼ùÿ?ã°¾{	¿|àn?×þ¿Û¥}?Ð¿Å°v?Â¸6¿YwÙ=|ò.?`ôz¿×¡R>²c?§ ¿sM¿% ? o?É]½úHt¿ã0¿`>¤q?MýX?Å(5>ª¿Cúw¿Ó«i¿J ¿¡h=)?1_d?  ?	h?·x*?2°>¬ºÒÎ¤¾C¿oöJ¿
l¿B}¿¦$¿Óþt¿¤a¿é¶G¿')¿`ï¿ãøÎ¾Î¾ñÙ¾T¿ø¼yJ¥=ª;>©¾>Ö0¹>Kà>É¤?C:?6?ÉE+?6?`??úÓG?bO?ÛxU?Û[?á_?Æd?Èg?ëûj?YÄm?s/p?3Ir?t?R±u?íw?ìAx?lJy?Ä/z?öz?ë¢{?G8|?¶¹|?Ý)}?	}?9ß}?'(~?Ug~?~?vÍ~?ö~??ã8?S?«j?­~???¬?D·?Á?tÉ?ÄÐ?×?Ü?Sá?på?ÿè?ì?Àî?ñ?ó?Ìô?Mö?÷?$7¿0z?Bæb¿ÒG?]S¿j²W¿Mg¼ä>L³è>¸8=]5¿J¿h¿¦ÿÑ>v<?»Îz¿Ji&?üR¾¬¼!>Ío¾½b ?I5Y¿uT~?*K¿L£Â¾Èã?&m¾Æ
e¿¦ó> j?.ën¾Ã¿:¡Ì¾Dy&?´í{?8Ò>Ûë¾\Üw¿B]¿4¾´>ùU?Wÿ?=QZ? Ñû>ü-=nëÅ¾e¢8¿Dm¿uà¿Í±t¿vR¿$!¿§Î¾¹¼+¾Ðòz=É>éÇé>rÔ?Q4=?APV?Àh?Du?~¬|?È?Õ[?¾|??v?ßgo?öf?	¥]?#ÂS?I?A:??ð4?Î*?4ë ?ÓX?ô"?©Q?Óù>Ûé>È»Ú>8rÌ>_ú¾> N²>3f¦>:>zÂ>Dõ>ç{>Woj>mkZ>9wK>]=>%}0>X$>O>ßx>n£>Ëóö=Ìáå=xûÕ=»,Ç=Àb¹=å¬=« =£v=c=tu=öp=5@`=²P=­7B=¾4=3(={=ª==Lü<'Éê<3}Ú<³RË<5½<5°<)Ú£<Fz<|ä<Ç
<3?åZ>qí¾ÿn=â[Q?9á	¿õ¿R1e¿7d¿Î½¿u4¿6Õ>éyi?A-¿2M¾B?¡Æs¿_É|?¯íx¿¸z]?|¿é½§aQ?ÑÇl¿
_ð<oÂs?Â²ä¾·'a¿LÏ>9ïx?¡ó/==ªj¿èzB¿âæ5>9mi?@8c?!> ¿%Ts¿Co¿¬¿ñI;%±? å^?{Å?vl?¥T1?ÿ?À>÷"þ<Fy¾n¿ÜëF¿±8j¿À_|¿ã¿^v¿pÁc¿[VJ¿>r,¿=¿Ó2Õ¾ú´¾$¾~\(½ªÞ=x92>¤z>Vµ>ïÕÜ>¢ ?Û?&Ö?03*?Æ5?°>?ÒG?wxN?|êT?Z?Èt_?h¼c?jvg?{´j?@m?ùo?Yr?lós?u?Oòv?]'x?e3y?Ìz?Gåz?ë{?G+|?r®|? }?}?å×}?Î!~?Õa~?L~?VÉ~?óò~?ý?56?>Q?¨h?ï|??µ?áª?I¶?*À?¸È?!Ð?Ö?Ü?éà?å?°è?Ðë?î?Üð?äò?¦ô?+ö?|÷?a O>8A?Nm¿r>?>ìÞ}¿»k¿Ü~¾g+¹½h®ë¾¬êt¿H"¿f]??Â×>Ò|¿¼]?Ä¿Úà>a±ù¾ÿ4?Gõr¿´Åq?½Ë¾+¿æ.}?ö¾ßs¿×°>»u?¼ªã½Ë|¿wÉý¾õ9?q?ûÎù>ØÆ¾Î]r¿"e¿ôT¾¾6B>Î¼M?	µ?ú`?bà?¶¬=÷=³¾|	2¿hÝi¿m¿cÍv¿~V¿*(&¿¬³Ù¾qøA¾uò&=î«>ä¤á>Òu?È:?øHT?ÒNg?*Vt?.|?ã§?ì?|?C/w?í p?ÕÈg?^?#°T?ÐJ?.@?â5?4»+?ÁÑ!?_7?ø?½?
Xû>ðKë>GÜ>»Í>0À>]q³>£w§>ê:>ª²>Ö>R8}>:øk>aÚ[>ÈÍL>Â>>N§1>n%>~	>mj>>/ø=thç=Qg×=È=Dº=¬±­=-©¡=Cu=k=R=Lr=¾a=êR=äC=Uò5=hR)==£=Úu=úý<¾Yì< òÛ<¢­Ì<tx¾<©?±<Ãñ¤<x~<Ö<ì<ü¶z?ï'¿±À>á*¿¶}?Ë>I¿g}¿ó~¿ôCc¿_¾}¾E?«*?Å(h¿&>Õáÿ>aO¿E1f?Y~_¿£ö5?V¡¾­O¨¾dk?uüU¿®|¾;P}?º¾Ä>p¿>Íi~?µ!>rV^¿Ï R¿>ó®=v_?¬ùk?Û¤>=â¾p§m¿Òt¿W¿ÛC½Ðõ>çöX??ÑÌo?
ô7?øBÐ>O9=`¾K»¿7ÀB¿¡´g¿~]{¿É¿T§w¿SÌe¿çL¿ÜV/¿¾¿cÛ¾^È¾10¾XTT½NÞx=RÕ(>û3>Sy±>]Ù>Ãý>4z??5)?À&4?Ú=?^F?ÔM?G[T?Z?È_?u]c?Ä#g?lj?ÃGm?5Ão?2ëq?}Ês?ju?}Óv?¢x?7y?³z?ÛÓz?Ñ{?1|?£|?F}?z}?Ð}?k~?M\~?~?/Å~?Zï~?à?3?çN?¢f?/{?û?d?½©?Mµ?P¿?ûÇ?}Ï?ýÕ?Û?à?¸ä?`è?ë?Iî?¨ð?·ò?ô?	ö?_÷?în?½ù5¡¾¦=û`F?Z-¿£Æw¿Lé-¿¿ö¤W¿JP{¿ fn¾ 3t?ÇV.=c]¿³{?z%O¿R÷-?h¢4¿æ\?¿#}Y?ÊM¾2¿È}r?í¿§<³í|¿ÆÑT>ü}?
P<tv¿¿@	ù>%þ?TÃ?Uà¾k¿ëêl¿NÆÜ¾âr>ÙD?·~?kHf?É?@K>ÙI ¾Í3+¿$1f¿þÜ~¿C²x¿Z?Z¿³+¿ú ä¾ûX¾*À¥<é]p>oÙ>p?ìÍ7?6R?Óe?­^s?¨{??x«?Øà|?Àw?ýÕp?Îh?.i_?ØU?&rK?0!A?ËÒ6?\§,?t·"?:?Í?\é?·Ûü>Â»ì>ItÝ>Ï>eÁ>W´>Ý¨>(;>¶¢>º¶>Ü~>ùm>7I]>?$N>@>gÑ2>&>¢>ó[>e>:ú=ïè=#ÓØ=bÒÉ=ÄÙ»=n×®=­º¢=ás=rô=¼.=,t=Þ<c=ÅyS=ÎD=&7=Nq*===]	=þ¨ÿ<Têí<ÌfÝ<Î<O»¿<l²<^	¦<«<ÃÈ<pÍ<§Ô·>cF¿úr?æÙ¿ÛÎ!?_<?Õ»¾ÍÛ;¿·YK¿Cö	¿n	C>4÷x?¡¤>Ä¿5?Ý:>°k¿ÑÎ;?Ig5¿æ±?U­ª½	¿ëÆz?½f7¿¤¾Bò?Ì%¾ iz¿Ü>·ú?Å> rO¿:­_¿­ö»7ÒS?3s?ÎµÈ>&øÁ¾Cúf¿Ï±x¿ª#¿}ËÌ½¨ß>#R?ó}?§!s?¦T>?aà>ÉÀ=¦Úr¾VÎ¿3t>¿6e¿ç;z¿ò¿êØx¿Äg¿jO¿Ö02¿º¿Má¾¼Õ¾Ô;¾ö"½èøO=nm>Äê>­>âÕ>ýù>Ü?c?Ý	(?v23?Ú=?_¢E?Ò/M?;ËS?×Y? ^?íýb?Ðf?B$j?àm?o?¾»q?J¡s?ÁFu?x´v?ºñw?ây?yóy?QÂz?u{?|?°|?b}?q}?É}?ü~?ºV~?­~?Á~?»ë~?½?Ë0?L?d?ky?s??¨?O´?t¾?<Ç?ØÎ?nÕ?#Û?à?[ä?è?Eë?î?tð?ò?Xô?èõ?B÷?EpN?ñ8X¿xÛ?Æ2¿¯r?£<½íCl¿Ùy¿}üo¿"»¿|HG¿Æ¨]>@~?õ|¯¾Ü` ¿á|?H§s¿Ýõ\?ö^¿+öu?fû|¿B¨6?mE;I0R¿Ã&`?íS=>öÿ¿R/=ÿõ?m£>©Õl¿gd+¿!Ë>Á´~?Z!?\hp¾STc¿zs¿gIú¾½\6>W;?w}?tk?®ô?  ,>¾®#$¿Ô@b¿Rù}¿`z¿PÔ]¿ÍÜ/¿_mï¾$n¾Í¹}M]>·(Ñ>?ç	5?	P?èLd?]r?ô{?ÃO?xÉ?ý<}?Mx?q?mci?G`??V?aL?OB?6Â7?w-?L#?cò?¢?´?¤^þ>ú*î>ÎÏÞ>©KÐ>¦Â>·µ>á©>9;>>J>O@>	o>ð·^>zO>AA>pû3>|'>¼>qM>F>×Ýû=¬uê=ï>Ú=.%Ë=?½=-ý¯=)Ì£=}r=wá=_=ËÆu=0»d=ÝT=MF=ÙZ8=2+=¨==aD
=¸« =êzï<ÛÞ<~cÏ<+þÀ<³<ø §<Ý<æº<Å®<d¿â	¿ÓZ?|y7¿k½vº?R Å>Z_¾>²¾fº;½f± ?.îy?èåî½C~p¿SG?¹h¾ÿ¾ßF?òû¾fø>¥Å>Ð]3¿´ÿ?µ#¿£R÷¾{?wºq¿y$<Í}?ã_Â>ã(>¿ýj¿Ç¦Í½øF?HØx?;oë>Yu ¾ÒS_¿`è{¿<x.¿Ù¾kùÈ>°ÆK?Ü\|?¦v?ItD?ï>º >¸zU¾bÿ¾:¿ÒHb¿ûx¿ÿÿ¿<óy¿ ªi¿ÞÞQ¿ÿÿ4¿¿G¥ç¾ÖÜ¤¾qG¾õ½/'=>&>y>¶©>cÒ>TÝö>ÿ³?'(?(ó&?ê<2?
,<?åD?L?Z:S?YY?Ð+^?Ðb?ð|f?yÛi?Él?Uo?ýq?Õws?Å"u?@v?¦Öw?fíx?ßy?«°z?Rf{?Â|?3|?n}?ßh}?Á}?~?Q~?Ò~?Ì¼~?è~??.?/J?b?¤w?ê?½?q§?O³?½?}Æ?2Î?ßÔ?¦Ú?¨ß?þã?¿ç?ÿê?Ðí?@ð?\ò?0ô?Åõ?$÷?. }½o¿rÃ|?~þ¿y/?@I?* ¿Km¿z¿_c¿é×É¾;>?Á[?|-¿Æ¾n"a?^ÿ¿@õx?¬x¿é?ç±l¿÷
?®ÑS>j¿ë½F?ÑA°>1}¿^
½H~?9>%`¿Ì>¿T2>7{?r2?¾	ßY¿x¿3_¿ñYò=ð<1?¦z?Ùp?$?µgV>GWs¾Û¿£^¿£Ô|¿BÖ{¿²>a¿ª4¿Lú¾¿¾Ü¬ª¼a(J>åÐÈ>[?æ:2?ðM?S¼b?ôRq?Ez?`?íà?}?ÍÔx?4r?°+j?0"a?UlW?OM?ÝC?W°8?|.?H$?ØÎ?v?7?Ñàÿ>ï>Ô*à>XÑ>nÏÃ>Ù¶>¯ªª>;>e>¾w>@>	p>&`>äÐP>B>j%5>]°(>Ì>ç>>'>ý=;üë=³ªÛ=ôwÌ=µP¾=è"±=£Ý¤=q=zÎ=ÿç=aw=9f=wAV=dG=9=¯,=³ ==£+=ñ=ñ<bPà<l¾Ð<AÂ<Å´<8¨<<	­<<>¿·>P">ßeÞ; f:¿EK?^]? À>9R>ÇEë>}Dk?qH?_P¿3¨<¿ÿ¦s?¹ó¾Ø;n>/?s¾¥Ö<ÑÃ>ÿV¿vz?NÏ¾§\!¿BZp?ðû><¿yÉì½cVw?/¨÷>'®*¿¹ôs¿E¾¶à7?à|?ñj?0U|¾`¼V¿3~¿&¹8¿íFP¾@Ð±>!D?ZSz?¦ªx?ÝPJ?òÇþ>¢ >vë7¾
Qó¾ë}5¿Úa_¿Kw¿Æñ¿1öz¿»|k¿DT¿+Ä7¿~¿µ¶í¾qÝª¾ÎS¾¬½_>ü<B>û¡p>Ð¥>òáÎ>zºó>N
?Aë?Û%?F1?S;?Å&D?oãK?£¨R?X?×¼]?=b?Á(f?<i?ìl?Ao?î[q?Ns?þt?Õuv?e»w?ÄÕx?Êy?çz?ìV{?jö{?£|?iø|?0`}?º}?ÿ~?zK~?î~?¸~?kä~?h
?N+?ÍG?{`?Ûu?]?e?H¦?N²?·¼?»Å?Í?NÔ?)Ú?;ß? ã?mç?¸ê?í?ð?.ò?ô?£õ?÷?`_¿3Õ¾3­B?0¿N®;Ãy?À¦û=°¿²ñ7¿à'
¿9§=íf?¢Á?Ë	h¿£MZ=¤+?Yôr¿H?Ú¿³Uy?å(O¿Õ±>MÔÍ>Úy¿h'?Üý>}	t¿S`¾x?ÈFÀ>LP¿(P¿6lS>úu?ÓYA?¸½?0O¿æ{¿K¿Ö/n=[&?w?æt?ZÃ,?6>!L¾Ö]¿ÏY¿<o{¿®}¿Ö}d¿9¿N¿þì¾hh)½_ð6>ÑhÀ>%?a/?X½K?k!a?Í>p?	áy?sÙ~?Ôñ?nä}?ÆWy?Ýr?ðj?}úa?QX?¶:N?ÖïC?-9?e/?gc%?ª?YI?rI	?± ?ñ>Zá>¡ÚÒ>âÅ>´û·>G»«>Ù: >r>X>ä>Xr>a>'R>Ø¿C>TO6>1Æ)>Ñ>T0>	>T$ÿ=Âí=pÝ=´ÊÍ='¿=H²=ï¥=­o=z»=Ä=Aüx=Í·g=M¥W=²¯H=ZÃ:=ùÍ-=¾!==æ=)Z=ò<-Åá<YÒ<áÃ<xñµ<+P©<B<,<oq<6mù¾°hv?q>&¿á9?ÿ¿¬`>]~?EàU?è2?0W?,%?YýÜ>#ÂQ¿ªGØ¾Û¢?cï=¿´[¡>)@r½d@=ã$h¾ûf?×3p¿gj?ÁÑd¾"ùA¿M^?¯>Ìy¿Û|¾+Cm?ûÑ?V;¿í{z¿É¾¯È'?¼C?Ú\?>6¾1=M¿¿dB¿7¾(;>Dê<?ç×w?¡Üz?cèO?Ü?.@>p3¾üãæ¾&Õ0¿»Z\¿v¿ìÇ¿°á{¿E<m¿GV¿.}:¿ö ¿R½ó¾P×°¾Z^¾ÿóÁ½êYª<d!>1h>iç¡>`]Ë>ð>Áç?î¬?²Á$?N0? y:?kgC?Ú;K?R?$X?6M]?ØÛa?Ôe?Hi?ÛIl?æn?+q?$s?Út?7Vv?øw?û½x?ÿµy?z?mG{?üè{?ÿt|?Sî|?sW}?²}?p~?ËE~?~?M´~?ºà~?5?(?gE?g^?t?Î??¥?J±?×»?ùÄ?âÌ?¼Ó?«Ù?ÎÞ?Aã?ç?qê?Uí?Öï? ò?áó?õ?éö?£¼a¿$?¢ÿº4C=Åz-¿êwc?ª®2?Yô·<-Âr¾
h?½"Þ	?tþ?-@>qÁ¿îÐ> Å>ZÑM¿Ó,p?÷s¿.§b?&¿%@>Æ?Ù¿ ?s!?:Ie¿ì4¶¾ÿ}m?Qù>·Ã=¿uÏ^¿ò­Ü=àÂm?@#O?©8;TWC¿x~¿C&¿-»ôb?nÖs?Êw?Ç$5?üþ>V$¾)­¿©ãT¿xÉy¿ ~¿g¿=¿Ñ~¿«À¾h}½C§#>*ñ·>& ?|,?`I?D|_?-!o?F8y?ý~?.ü?¶/~?öÕy?s?²k?øÏb?3Y?c$O?:ÜD?¶:?rM0?§E&?¤??5
?rq?ótò>aßâ>!Ô>8Æ>¡¹>¨Ë¬>f:¡>a>P8>Ûµ>¢s>ec>)}S>ÿD>.y7>øÛ*>Ì>¹!>é	>Ác >?	ï='Þ=nÏ=ÇÀ=Sn³= §=An=y¨=<¡=øz=6i="	Y=âúI=÷;=Üì.=É"=}='ú=a1=©,ô<÷9ã<FtÓ<»ÆÄ<ì·<Ågª<t<O<ÃR<i|ñ>0ÞJ?àÿ¿µ?äA<¿åê¾S7?zï?Á³x?h¸?d´W? á»s{¿ì2½Qèi?ÞHl¿Ç;?à8±¾#>Oî¾ÿÛB?ÖÁ}¿ö}P?+Q½rw\¿»«F?ä·ã>§=o¿l#¿¾Ð_?dÔ+?ü¾m~¿²Ë½¾Án?Aÿ?gv%?µ£ß½àB¿Öÿ¿¦rK¿Mð¾sH>å4?¥ët?º«|?ð8U?--?;`>v²ø½JÚ¾û,¿ã3Y¿<t¿v¿£µ|¿èn¿ÿâX¿à*=¿Iè¿Õ¸ù¾6Ê¶¾Lj¾¾Ù×½2â0<<Yó=ô[_>
ü>áÕÇ>²kí>i?0m?ö¦#?ÏT/?Ç9?§B?WJ?¹Q?oW?îÜ\?ÿya?Ý~e?kþh?f	l?®n?êúp?àùr?sµt?f6v?_w?¦x?>¡y?{z?Ô7{?wÛ{?Hi|?-ä|?¨N}?éª}?Öú}?@~?|~?°~?Ý~?ý?À%?ýB?P\?>r?=?°?ï£?F°?õº?6Ä?9Ì?)Ó?,Ù?`Þ?ââ?Éæ?*ê?í? ï?Òñ?¸ó?^õ?Ëö?Yê¢½ß÷?ÔRC¿¾ë@?)A¿Æ§Å>^+}?&?)¢>Ýê>­Ç^?ßbe?>Xl¾dp¿Î´4? b=.E¿P&L?¼åU¿U=?6ç¾?i´½]ñ;?¯Û|¿"²>lø??ÉQ¿ó«ø¾S­^?³y?¿³(¿·k¿x<êc?G_[?®=Ue6¿Ì¿N2¿oF½Ì´?ñlo?jsz?ý&=?©>¨ù½Ì¿ïO¿Ããw¿þè~¿òwj¿,áA¿¿n¢¾7¦¨½ÚN>j¯>d?§)?Ð6G?ðÌ]?!úm? x? E~?ûÿ?[u~?YOz?#t?5pl?¢c?Z?!P?ÇE?ðr;?N41?''?ù_?Kî?Ü
?e1?´áó>ç8ä>ùgÕ>ÐkÇ>J?º>ÑÛ­>Ç9¢>âP>m>>*u>¥qd>&ÓT>M>F>ø¢8>²ñ+>½ >>vÊ
>S5>´ð=Öíß=#pÐ=þÂ=´=þ¨=Ól=v=Ø}=­1|=c´j=õlZ=FK=Ö+==½0=Ô#=v=iá==<½õ<Á®ä<3ÏÔ<	Æ<^J¸<^«<¥<r<4<Q0?	¼¸{%¿6G(?ö-½¡'l¿îÚ>ôWO?Ãr?zc?;ü>Pã¾y¿øò®>õT5?*¿æ°P?v¿¿§?BN,¿ôhd?9¿ÿÑ-?ð>°p¿\)?I®?,Ç_¿ü¾Z`N?Ü@?ÐßÊ¾ìý¿U)é¾7ô?o?Þ¡3?~L"½±7¿S¿ÜS¿Ë;µ¾VT>Ü,?èq?;~?µ@Z?ÑS?µ>ëÆ¼½VÍ¾6,'¿ÆíU¿cÃr¿j!¿öq}¿p¿[¿Í?¿NÇ ¿û¨ÿ¾çµ¼¾@u¾¹í½ÀP:Kjà=l²V>>KÄ>Ø?ê>?	,?ç"?NZ.?nÁ8?¤åA?èéI?îP?üW?ÿk\?a?')e?×³h?Èk?3vn?õÉp?]Ïr?t?bv?hw?õx?[y?íhz?"({?ÝÍ{?]|?öÙ|?ÎE}?>£}?2ô}?R:~?w~?±«~?EÙ~?À ?ñ"?@?5Z?lp?©?R?À¢??¯?º?qÃ?Ë?Ò?¬Ø?ñÝ?â?væ?âé?Ùì?jï?£ñ?ó?;õ?¬ö?!»K?â¬?o|¿'ÿ~?OH¿æ¾b`?ê]q?2-F?_eW?	æ?Zy?*Ý¿ÎÙ<¿ìi? Ã¾£	¾¤?Àã'¿È?Új¾µ¤¾&p[?0©p¿Ó4>?Y?jû7¿ù¿pèK?ÝÚ/?d¿Ét¿©9¼½]X?oöe?Ä#)>ãl(¿8â¿zÙ=¿{û½U?¦[j?¼|?ÇÅD?öº½>í·©½Ózû¾
¾J¿¾u¿{~¿Ä1m¿F¿ú£¿ï-­¾-Ò½èÑù=àÕ¦>Ï¿ ?j&?ÈãD?\?²Él??Íw?ï}?9ý?]µ~?ïÃz?å¿t?ì*m?prd?RñZ?ìñP?7°F?Ú[<?2?(?9?å¿?P¥?ôð?ÖMõ>ìå>	®Ö>GÈ>¯`»>Ãë®>ü8£>@>mø>Y>d²v>Åße>)V>i}G>²Ì9>_->£!>j>\«>ß> ò=~Yá=ÒÂÑ=b>Ã=°¹µ=k#©=ck=q=rZ=_Ì}=ª2l=ÆÐ[=>L=`>=*1=ß$=n=©È=Ïß=ÏM÷<#æ<*Ö<oLÇ<Ñv¹<÷¬<× <u<l<à?¯O¿äA&>ò#µ½¯?tt¿ù´õ¾wª>C"?vY
?5æ<ÕJ¿K¿XÕ,?~¯Ñ>s³u¿æst?+	O¿RBA?ãwV¿-y?âs¿'Ø?ñ®>ú{¿Ûl?Ý2?KºK¿AÈ¿Z
:?¹´R?çH¾ ê~¿:A	¿	ùà>Ï{|?ÚÊ@?\ñö<Y¼+¿~¿
[¿CÎ¾	#>óÁ#?6Æm??ûý^?RN?Jw>±½À¾§-"¿ßR¿Fép¿Ô¤~¿~¿"r¿«D]¿§cB¿ß#¿¾Æ¿(Â¾é¾¸È¾uÊ¼avÍ=ÆN>}>N¾À>ç>Xª?~é?m!?^-?öã7?8#A??I?YP?ÎV?iú[?´`?ïÒd?Ðhh?Mk?=n?³p?¤r?mkt?*öu?¦Lw?¸ux?Wwy?´Vz?W{?-À{?¢Q|?®Ï|?æ<}?}?í}?4~?r~?Y§~?Õ~?}ý~? ?>?X?n??ò?¡?7®?-¹?«Â?ãÊ?Ò?+Ø?Ý?!â?"æ?é?ì?4ï?tñ?gó?õ?ö?]p?I¾² ¿f@ ?U¾RW¿/Ñ>w?ê~? µ?e?çO>@Ua¿½ÌØ¾e?«¿­]q<÷¨>ª½Ú¾]i¢>Ã»;TQ¿Q¶q?¸³[¿Þ¡¹?l?X¿l6¿¨5?¨E?ìJð¾E{¿opC¾ôgJ?fÔn?7ãy>¿
¹~¿ÿH¿Ùâ:¾Ä
î>¦d?>k~?%ýK?Ñ>ç3½ë¾PE¿Zs¿UÛ¿¾o¿í$J¿x¿ÙÄ·¾jOü½ÂîÒ= 3>ò#ú>#?eB?PZ?îk?w?}}?ëó?¹ï~?´3{?²Xu?8âm?h?e?¯Ì[?ÄÕQ?ÍG?qC=?Ëþ2?#ç(?|?è?§m?!°?W¹ö>nêæ>¯ó×>iÒÉ>Î¼>~û¯>8¤>*/>PØ>*>:x>ÇMg>Ö~W>r¼H>\ö:> .>!">µõ>;>fØ>ó=Åâ={Ó=ÁyÄ=Xß¶=Ö4ª=ði=jo=7=g=ð°m=4]=jÜM=M?=}I2=~ê%=g=ê¯=·=bÞø<Sç<
×<IÈ<D£º<®­<	 ¡<·g<Àö<z[¯¾¢t¿ÄW[?f£G¿è9}?½v
¿µ¨i¿à¾Wû=¤C= Ââ¾«z¿qýò¾¼êg?ëoh=Ù°O¿ãø?Ïq¿øug?jÇr¿íþ?¿ï\¿ù§¨>ng?ÿÿ¿ÖÃ>óL??3¿ä4¿â×"?b?bæC¾JK{¿»½¿Z¸>®Bx?mÞL?ÿQÌ=Ï¿N³{¿F¦b¿TDæ¾²Cã=÷¬?Ki?^Á?"oc?#?ï>;ÿ½æ³¾#¿¨O¿"ñn¿À~¿z£~¿ys¿P^_¿kîD¿×k&¿³¿¼vÈ¾Ï?¾±¾N¼é}º=+SE>p*>U.½>mßã>¨=?¥?ØN ?£a,?`7?É_@?IH?©ÃO?äV?-[?üP`?5|d?Xh?«Ek?sn?$gp?yr?Ft?ÀÕu?0w?U]x?1by?_Dz?r{?g²{?²E|?VÅ|?ð3}?Ã}?Èæ}?´.~?m~?ù¢~?¸Ñ~?5ú~?G?§;?õU?¾l?y??\ ?-­?G¸?ãÁ?6Ê?lÑ?ª×?Ý?Àá?Îå?Qé?[ì?ýî?Eñ?>ó?ôô?oö?±`>·×q¿ì£>+0¾\?õ}¿çôg¾M%?³g?c?É?
|¾hR¿<7½ìs?VR¿ `¦>×$=Í*¾`=3Bv>¶.¿BÔ}?¾>¿ár5¾¢vy?aºò¾ùIN¿Âç?¾*X?qº¾4¿h¾)÷:?)èu?¼¤>gº	¿qR|¿
~R¿AYw¾60Ô>LP^?	?JÉR?§å>Û»DÚ¾Ä¨?¿.¸p¿xÿ¿Rr¿½N¿Iu¿ÛDÂ¾µþ¾ä÷«=>¾¶ò>É ?É@?±X?áLj?a@v?û-}?ä?n$?§{?líu?n?	f?¬¥\?¥·R?Å}H?µ)>?iâ3?ÝÅ)?¨ê?Ta?5?ên?6$ø>mBè>ê8Ù>6Ë>¨¢½> ±>Ü6¥>>¸>øû>­Áy>ª»h>ÔX>fûI>õ<>2/>P$#>øæ>m>ç©>Ý"õ=¹0ä=hÔ=µÅ=ý¸==F«=zh=b\=¢=Þ=3/o=d^='O=È@=\h3={õ&=_=*=<=ônú<é<ößØ<"ÒÉ<¶Ï»<)Æ®<:¤¢<ÙY<Ø<ØÂy¿è§¾¥©r?§Û}¿êR?ù
>}Xy¿HzC¿d¸Ù¾_tê¾ÏO¿x¿?÷½2¾?ih¾Cí¿jr?èÊ¿åi|?¼h¿­|x?ÞE;¿ >³½*?îò{¿ìe>Qga?,¿òK¿F$	?dn?Çð­½,)u¿ß.¿VY>mr?6ËW?ô,> ³¿Ykx¿ûh¿ßý¾¡Ö=®F?ðd?Xÿ?¨g?´¶)?0>®>=è»ºZ¦¾à¿¤dK¿4Ûl¿@Y}¿¿J×t¿Lha¿;mG¿1)¿B¿jKÎ¾ò¾8¾Ö3ï¼N§=Æ<>ô4>¢¹>ïªà>Ï?D`?Ý.?}c+?®%6?W??èG?ÿ,O??U?J[?Ôì_?ú$d?mÑg?¤k?Ëm?H5p?CNr?} t?#µu?<w?ÊDx?éLy?ì1z?tøz?¤{?¯9|?íº|?ë*}?ó}?à}?×(~?íg~?~?éÍ~?èö~?k?-9?ÐS?âj?Ý~?+?'?"¬?_·?Á?É?ÕÐ?(×?¡Ü?_á?zå?é?ì?Æî?ñ?ó?Ðô?Pö?ÿÐ3¿OÌS¿ô­m?N¿ïëy?!Ý-¿¸öD¿0­Ç=û	?
?Êè>ãH%¿p¿ìh®>èH?»òv¿ Z?¶©¾5Ù=?>¾ánì>MS¿G?Å×¿åm²¾.?sµª¾éÒa¿Vÿ>¿zg?§k¾ï¿ÝÃ¾1è)?%${? Ë>Yò¾j±x¿î[¿?t¾î©¹>®^W?Áù?¥&Y?7ø>`Y=6É¾QÈ9¿3Øm¿Üê¿Lt¿KâQ¿¹< ¿¥¬Ì¾ÖÅ'¾òð=iÉ>8ë>Ãl?­=?u«V? i?Smu?ýÁ|?¥Í?{S?Å|?~v?Fo?ÃÐf?G|]?S?bI?¤??ïÄ4?²£*?Â ?)1?äü?N-?rù>èé>»}Ú>¬7Ì><Ã¾>J²>5¦>à>¼>DÍ>I{>n)j>"*Z>E:K>~I=>H0>'$>2Ø>åM>b{>.©ö=Kå=¼ºÕ=rðÆ=*¹=¢W¬=g =WI=8ð=3N=t­p=0ü_=¾rP=ÁüA=:4=w (=ûW=i~=re=ÿû<ãê<á:Ú<ûË<(ü¼<ÁÝ¯<k¨£<üK<h¹<ß66¿Ì?b4¾>õ¿BÑ]>ç;?#¿ÄÇ~¿é¡W¿EW¿}¿Ù}C¿8¯>q°p?°¿)ò¾z;L?ýÈw¿^~?{¿ôc?×¿w½¨ÜK?óo¿å y=Zq?)ñ¾ú]¿p©Ú>(w?·³·<øl¿ç|?¿F>§k?}a?èr>4¼¿;;t¿Pn¿ác
¿W*b<	?ç_?iØ?!gk? 0?±`½>§Ð<¾¡¿Y¦G¿À§j¿f|¿Èu¿²!v¿{bc¿ðßI¿bí+¿Ay¿öÔ¾©¾ét"¾ ½ü=Ãä3>=>C¶>¤sÝ>` ???$d*?áD5?âÕ>?;G?N?àU?Â¡Z?_?=Íc?g?9Áj?Tm? p?´"r?­ús?Su?Å÷v?,x?7y?\z?]èz?{?-|?t°|?Ø!}?}?4Ù}?ð"~?Ñb~?%~?Ê~?ó~??¯6?¨Q?i??}?Ä?ñ?«?v¶?QÀ?ÚÈ?>Ð?¤Ö?/Ü?üà?$å?¾è?Üë?î?åð?ìò?¬ô?1ö?¹{z¿@3½qob?K|¿®­\?ºG½ìÁ¿ýqø¾Øá;4ÃF=£¯¾£0j¿ ®7¿&,?Þ?öP¿6R?Do¿õ½>ÞÚ¾t@(?ÒÏm¿Ì v?â¾Ø@¿~?{L=¾¸ºp¿ÌÁÁ>a;s?@¾õ×}¿Káñ¾Ô`?N~~?
.ð> àÏ¾}Ús¿ðc¿t¬¶¾b>ÖO?Ø?Ü_?±o?k=lã·¾ò°3¿W»j¿¿Mv¿öU¿í$¿ìúÖ¾Ûz<¾%»;=Ú>Ï©ã>/L?e1;?rÊT?«g?ät?N|?¯°?ß|?f|?
w?óo? g?P^?}uT?ÕDJ?<ò??]¦5?¢+?Ð!?e ?ÉÃ?Në?øú>ßðê>!ÂÛ>ËiÍ>ã¿>[)³>4§>û>Ew>w>[Ð|>k>¢[>yL>ör>>]1>Ò)%>dÉ>°.>ØL>v/ø=Öç=T×=Ä+È=;Pº=i­=e¡=J6=ÌÌ==´+r=ú_a=æ½Q=ù0C=¦5=r)=tP=¨e=¨<=ý<ªöë<ËÛ<ÓWÌ<(¾<Zõ°<¬¤<><¼<pS>7Á?!Ûî¾Ì->ñÁ¿²?©=2½D×_¿rþ¿Ë²¿çp¿ËÎ¾¾Q2?Y=?Ô\¿(=?yZ¿ºm?0ng¿¢ñ@?Å½¾­®¾e?Þb\¿' Õ½t{?#0®¾	öl¿l­>k}?¡>¡¡a¿rN¿ÝÝ=Àb?Gói?éÖ>>mê¾Ü&o¿xjs¿Bw¿Çã½B4û>DyZ?¢L?Aën?zV6?¸RÌ>w`=$¾c.¿QËC¿Wh¿J {¿!»¿,Xw¿¹Le¿cFL¿« .¿çR¿&ÜÙ¾:H¾ÌO-¾útI½`}=K(+>÷B>Fn²>9Ú>PÞý>Ñ?ë?c)?ùb4?m>?F?9ýM?Æ~T?-Z?Í"_?ÿtc?A8g?k~j?AWm?«Ðo?ãöq?£Ôs?Psu?!Ûv?Bx?ö!y?¯z?,Øz?{?o!|?ê¥|?¶}?-|}?YÒ}? ~?¬]~?±~?7Æ~?>ð~?¦?-4?|O?"g?{?[?¸?ª?µ?¿?*È?¦Ï?!Ö?½Û?à?Ïä?tè?ë?Wî?µð?Âò?ô?ö?É¶µ¾¹iF?Áó^>h¿>è»?¸oR¿Ûge¿Äþ¿ê¾Àì=¿5Ò¿ª3½¾Ëg?9*>È{j¿Ó9u?ä?¿åº?d=$¿§ÞP?ÿÉ|¿cb?G¾/'¿ºv?Ì½,³z¿>_>{?»õÙ¼²Òx¿Ü¿?7ð?>æ	?=¬¾°Óm¿W·j¿ñ9Ó¾ð>á»G?.?Ôd?$?§mç=üQ¦¾vd-¿dbg¿¿x¿ Y¿¹)¿f.á¾Q¾´Û<8cv>#Ü>="?ß«8?ÀßR?Lf?®s?Ó{?.? ?|Â|?	w?p?Vh?P"_?oQU?é%K?{Ô@?°6?«\,?Én"?Ï?1?é¨?ÿ`ü>QGì>Ý>Î>Á>38´>V2¨>ê>°V>o>wW~>m>Õ\>Ç·M>^?>ür2>,&>º>s>H>µµù=Zsè=æ_Ø=gÉ=Ôu»=az®=d¢=;#=^©=Ùè=ñ©s=ÃÃb=	S=0eD=óÄ6=m*=íH=æL=Ý	=¦ ÿ<qkí<¶ðÜ<¬Í<U¿<ò²<Í°¥<@0<|<¢Uo?#Ä!?ÇÛy¿¯T?[u¿ÏñK?NÈ?^<ã¾Y¿°c¿§+¿Ü=)ám?¾QÙ>q|¿ÔuÍ>ù>a*¿]/K?Ë^D¿³?­!¾ï¾çv?·ÞA¿¥¾Þ?(JO¾í¼w¿tyD>Íè?ÒÄp>'nT¿õ[¿É³<5¯W?nq?]w½>eÌ¾þ2i¿|w¿á! ¿µýª½j¼æ>x§T?:\~?Òr?oV<?rÛ>TZ¬=º|¾­±¿Ô?¿`ée¿z¿è¿¡zx¿å&g¿p N¿ÆJ1¿&¿¾ß¾ ë¾%8¾K[r½Êí\=h">F>·Ó®>ÑüÖ>Êùú>D?:Ç?Üa(?ú3?øG=?ÞE?dM?óøS?Á¸Y?ï¼^?@c?ëf?8;j?Öm?éo?ÎÊq?^®s?Ru?Q¾v?Eúw?Jy?åùy?ãÇz?qz{?3|?P|?}?7t}?tË}?~?X~?5~?UÂ~?áì~?¼?§1?LM?>e?úy?ð?}?õ¨?´?¹¾?yÇ?Ï?Õ?JÛ?6à?yä?)è?[ë?î?ð?ò?dô?òõ?M?ÞAx?5¿%±V>5ã¾ëy?`æ¾U}¿¿-f¿%W¿ï+x¿ÅÏa¿Ø0=Ýº?9ºI¾&Õ:¿fë?PEh¿àL?àoP¿
Ìm?¿ûBE?1½j×F¿Øh?Xù=ò¿Òø=-d?sÃµ=îp¿öZ#¿ Ý>w?WÐ?§¡¾¤f¿rÚp¿ï¾ZÓM>©??iÃ}?®i?¸]?>Æ¾¹ä&¿7Îc¿÷X~¿Ày¿5w\¿ò.¿ÍEë¾¡¥e¾ú;É¬d>]Ô>ï?¤6?tëP?Ôäd?Âr?=Q{?"c?©¾?}?Zx?%Cq?0i?¸ñ_?b+V?XL?aµA?çe7?Ì7-?D#??P?f?NÉý><í>¨IÞ>ÍÏ>Q#Â>ÑFµ>x0©>ZØ>ý5>@>kÞ>úqn>T*^>iöN>µÅ@>Y3>(/'>­«>/ð>²ï>ê;û=ÖÞé=q²Ù=X¢Ê=j¼=¼¯=b£=*=ï=*¶=,(u='d=2TT=fE=Îã7=g!+=eA=$4=ë	=X =7àî<KÞ<ÝÎ<|À<$³<ý´¦<a"<c]<;ÄM?$ðy¾sWL¿kOz?Te¿ñ`c>õ[s?hd>Qà¾¼
¿ºK{¾Ñ4ñ>äÂ?à<=»ûz¿X/??dÍ¼G×¾í? ¿½½>K~=+#¿WA?;=!¿ú×¾u~?¶Ðw½¯~¿	%=aý~?­>OE¿Ìäf¿N½MâK?±Úv?¸+Þ>]­¾.eb¿yÆz¿E\*¿ß¾JËÑ>átN?}?¿ýt?¹B?é>¤Rè=ÅØ`¾i¿CÁ;¿_c¿¯zy¿þ¿øy¿Üðh¿ñíP¿ë3¿¡ò¿Jå¾¬¢¾âõB¾´½ïÛ6=«¥>|>¤6«>]½Ó>ø>=?&¢?ó^'?ã2?<?O.E?4ÊL?frS?ICY?~V^?ÿÂb?Of?¢÷i?âl?Ûjo?wq?às?±0u?U¡v? áw?|öx?ýæy?·z?=l{?ä|?¥|?H}?5l}?Ä}?~?JS~?³~?m¾~?é~?Î?/?K?Vc?Tx??A?ã§?²³?ì½?ÇÆ?rÎ?Õ?×Ú?Òß?"ä?Þç?ë?çí?Tð?nò??ô?Óõ?¤o?bÄ>u¿ ¿Y?o¿/Éc?6Â§>f;¿­x~¿è¯¿´¸x¿ 8¿*å>jÉp?Q¼¿| ë¾Ù;q?¦}¿ªÀn?û+o¿!j}?ÕÔu¿SØ?ßê=M:`¿7&S?0>®¿ñÝC¼w?BÌO>ÚBf¿76¿ýA±>Ý}?7¹*?+D¾òU^¿¢ùu¿_õ¿>é5?Ñ{?Én?eï?ï C>Ü¾§3 ¿ºÿ_¿b}¿¦2{¿£µ_¿m2¿á?õ¾ðz¾<¼×ãR> Ì>³?×3?¦íN?tc?Íq?qÇz?2?×?Ël}?x?Àåq?ÝÐi?µ¾`?VW? ãL?êB?D8?.?$?}j??ì"?÷0ÿ> òî>Çß>þÐ>ÉBÃ>5U¶>k.ª>Æ>+>~>²>>ßo>_>õ4P>ûîA>©4>Ã1(>Å>åÐ>Á>Âü=KJë=÷Û=ÝË=ûÀ½=°=
a¤=ý=~b=z=e¦v=Pe=VU=ÍF=©9=`,,=Ý9 =b=FÂ
=â =ýTð<¦ß<[ Ð<í­Á<!<´<.¹§<<·><©Û½ikl¿1àn½Ð?´¾¸©é¾GÞq?9C.?yß=¹pJ½
qr>âJO?áÔd?ÒÞ­¾lX¿Gc?¿_«¾T¾'À¸>¡¶¾H*>¡Þ>ÄöG¿¤P~?Ã÷¾|¿¿×Nv?ßM©=Qû¿©a½q¬z?¿ß>KÐ3¿%+p¿ÔH¾£Â>?À={?8Ëý>fà¾ÊÃZ¿]E}¿B4¿éP8¾åk¼>PäG?&O{?w? ­G?óß÷>º>¨hE¾ëø¾g7¿_¸`¿m?x¿¯û¿z¿ªj¿Ä.S¿ã6¿t¸¿Eôê¾*"¨¾qÀM¾"
¢½Æ=Ùß>Ðt>§>E{Ð>¥(õ>¢ñ
?Ñ{?ÝZ&?·¶1?¶;?}D?{/L?!ëR?-ÍX?{ï]??ib?,Of?¨³i?û¦l?7o?Ýqq?&as?u?,v?ÕÇw?àx?ùÓy?§z?ó]{?ü{?ê|?ûü|?&d}?½}?ø
~?N~?)~?º~?æ~?Û?,?ãH?la?«v???Ï¦?Ã²?½?Æ?×Í?Ô?bÚ?mß?Ëã?ç?Ùê?®í?#ð?Cò?ô?³õ?þq÷>ñ¿2ô0¿4èw?Òl¿CïÆ>	U?t;]¾:dH¿÷Êc¿Sr?¿61¾Û`F?Ô<=?ÑM¿|G¾ J?ò|¿tÖ~?u~¿$Ù~?á;`¿Clç>£/>¾r¿»k8?ÀtÕ>|y¿E¾`æ{?ù >ïX¿BG¿CÛ>Ëx?m9?\ñ½OòT¿^z¿yí¿Fä·=Þ=,?ýFy?Èr?F8(?kjj>ÈÂ`¾:S¿æ÷[¿ï2|¿yt|¿âÎb¿º6¿bÿ¾ 6¾`ú¼Ó	A>»ÔÄ>!n?á0?læL?qúa?ùÐp?;6z?nû~?Åé?¦º}?y?ßr? j?Fa?FÙW?@¿M?sC?ý 9?Që.?Cì$?N7?vÚ?Sß?üK ?}Gð>yÏà>×.Ò>ùaÄ>_c·>.,«>´>;ô>Oâ>îu>bLq>Ô`>msQ>0C>ë²5>S4)>Ó>±>v>9Hþ=¸µì=wWÜ=ÙÍ=æ¾=i®±=_¥=ê=?=ÉP=$x=ïf=yêV=ÎH=!:=Y7-=T2!==z=)é=ÂÉñ<rá<3cÑ<]ÚÂ<¹Sµ<^½¨<¥<
 <`¿Yú[¿×þ8?p¾ñnÂ>Øâk¿¤ê?<ôy?O?Å¢é>ô)?'#|?Ï!?æh,¿ÐR¿B}?>u¿Tª>nïÂ=^à½ÚÂ½÷>\[d¿¸t?cÒ£¾}1¿é°h?Ëe>YM}¿ó6¾³s?Øì?W¹ ¿t]w¿bBq¾f0?J7~?;?¨N[¾ôUR¿G÷~¿ d=¿!i¾v©¦>Äø@?¯3y?ØÁy?M?.õ?ÂÍ/>¥Ò)¾Ùoí¾K3¿¯õ]¿^év¿Wá¿ùh{¿®Sl¿ÄbU¿9¿jw¿Àð¾ì´­¾êX¾jr¶½ËYÕ<?>wl>%õ£>6Í><ò>\¤	?>T?U%?wÐ0?¦ë:?ÌC?ôK?#cR?mVX?ç]?ýb? f?Koi?kl?Úo? Eq?3:s?Hít?×fv?d®w?}Êx?×Ày?mz?O{?ð{?{|? ó|?\}?¶}?ã~?ÈH~?~?¶~?«â~?ã?þ)?ªF?_? u? ?Â?º¥?Ó±?M¼?_Å?;Í?	Ô?íÙ?ß?sã?Fç?ê?uí?òï?ò?öó?õ?P{ó¾¢~¿wÒ=×6û>¤Ö¾µ¾ó?]Á>~0©¾î
¿DS³¾>³:w?®ÖÙ>/>v¿ÚL>ì?:¤f¿î{?½O}¿r?ÞÂ?¿Tn>ô>ß4}¿W§?(?
¼n¿Õ¾Pt?%ç×>¥I¿ìNV¿ß *>¥r?cG?»¢0½PJ¿@}¿2]¿ë	="?%v?u?40?À«>#<¾zE¿Ã·W¿îËz¿Ö}¿lÂe¿Ìí:¿k¿/S¾{K½. />û¼>¨ 
?6.?ÞÕJ?Üw`?Ìo? y?È½~?Ñö?£~?y? s?w?k?gQb?2­X?µN?äOD?Úü9?³Ã/?B¿%??ä?R	?)ÿ ?Òñ>½â><_Ó>áÅ>Oq¸>Ã)¬>¢ >,Ó>³>,9>f¹r>)b>Ð±R>TAD> È6>Ø6*>Ù~>:>Ïc	>SÎÿ=!î=ð©Ý=TÎ=À=»¿²=ÿ]¦=ëÖ===Ð¢y=ÖRh=5X= 6I=[@;=QB.=Ë*"=Üé=®p=p±=>ó<Z\â<
¦Ò<ÎÄ<Pk¶<Á©<Æø<^<F3a¿1Ó½'¦~?"_¿hh?¨t¿ê|½ám?vq?§W?ÖÈo?Èv?ûÞ>g¬g¿¾4Óz?¶S¿ó,Þ>UÔ5¾>ðä¦¾6)?e'w¿a?Ü¾ M¿+U?Ø¸>ôv¿zì¾B h?}h?F¿Èk|¿6£¾Qä ?	Ã?Ê?Øï¾#I¿×Ú¿$F¿à±¾o>aµ9?ýµv?d¤{?ÊR?ÜØ	?ÒhM>	¾4Íá¾é.¿T[¿¨xu¿¯¿{:|¿Lìm¿ÑW¿ ;¿d/¿¿+ö¾¿A³¾ýBc¾ÖÊ½Ò"<þ=/d>ÕP >TïÉ>êLï>ÍU?o+?3O$?$é/?< :?C? ÷J?nÚQ?	ßW?Â]?<´a?±e?*i?Ã/l?çÏn?áq?s?GËt?VIv?Ìw?K´x?­y?¾z?A{?ã{?Ap|?7ê|?äS}?r¯}?Äþ}?{C~?~?²~?9ß~?ç?h'?mD?]?Rs?,??¢¤?á°?{»?ªÄ?Ì?Ó?wÙ?¡Þ?ã?úæ?Tê?<í?Àï?íñ?Ðó?rõ?ÀF¿°H-¿%ýR?dá¾ÐV >KòV¿òA?#V?Üe>4N½Þ>%,+?}Ã|?t­@=F¿§ã?ä>´<¿^@f?Ík¿'¤W?ùü¿¨n='"?â¿vné>o>,?®_¿7Í¾Áøh?{ú?Aô6¿7c¿x"=e­j?mS?´q=ô?¿¿ë8*¿üÃ¸¼?»mr?;x?²à7?lì><C¾}¿h@S¿U-y¿f~¿Åh¿?¿ã8	¿`¾îÎ½\(>;µ>ÌÊ?]+?¼H?mì^?¿n?¤ýx?y~?0þ?¿G~?Kÿy?¡¸s?^òk?c?Y?~rO?R+E?×:?)0?&?Ï?Òb?êV
?²?ïò>Sã>GÔ>Æ>¹>''­>A¡>ý±>¦>Vü>H&t>~c>ðS>gjE>FÝ7>R9+>Õo>Ùr>"5
>2ª >|ï=düÞ=FÏ=1Á=	Ñ³=u\§=ÑÃ=!ø=bë=!{=¶i=ºY=2jJ=2_<=HM/=@##=Ñ=áG=¶y=K³ô<C·ã<àèÓ<>3Å<ç·<¾Åª<çê<±â<óà½þo<?Sû?çu¿s?þ¿'¿I?Ýy?ø¬?SÞ}?[>?O"¾s·¿d´==i[?ùu¿k5-?ÀÉß¾çSÇ>÷	¿²uO?©¿=ÀE?hmõ<_Øc¿|c=?$#û>Ãj¿e:Ô¾C$Z?K3?Þë¾îJ¿ÁÌ¾IW?ÂÞ?5U*?ÑÕ¯½'5?¿Tï¿$YN¿Ð}¤¾ÖRt>w2?×s?1}?øñV?A?YÖj>Jä½Ö¾¾m*¿©X¿rís¿ëd¿÷|¿=to¿È£Y¿Å>¿Bà¿
¹û¾tÈ¸¾Xúm¾r4ß½:£ó;¿üì=s\>6ª>¥Æ>([ì>÷?g?¢G#?À /?ØS9?'fB?ZJ?QQ?gW?·\?úXa?be?gåh?¥ók?¨n?~êp?ër?©t?©+v?{w?÷x?<y?õtz?2{?èÖ{?Te|?¿à|?°K}?W¨}?ø}?&>~?bz~?®~?ÂÛ~?ç?Ï$?,B?[?¢q?µ?;?£?î¯?©º?ôÃ? Ì?øÒ?Ù?;Þ?Ââ?­æ?ê?í?ï?Âñ?«ó?Rõ?ª¿j>>ï"w?Ðd¿F^`?¯~¿àµU>ì?òx5?æ9é>î?Y<m?þV?ªT­¾H#f¿{E?
½O*¿i??K¿ï11?²áÉ¾y
¾Õ,E??~z¿->FAG?ÞJ¿mË¿¾Z?8?±"¿Ûm¿P7¼ó`?S\^?8eÚ=qÄ2¿ñ¿u5¿'2¡½Z?B#n?>{?9??¢ï®>p_ä½gª¿ùN¿Ww¿d¿u6k¿ÚC¿ õ¿(\¥¾îÒ³½Ñ#>*"­>Âl?¦Ã(?'F?5X]?Ü©m?KVx?ï.~?ãÿ?ú~?énz?@Mt?U¢l?RÚc?ôNZ?IP?]F?.±;?°q1?ûb'???&??d?ßBô>öä>÷¾Õ>Ö½Ç>}º>\$®>Ú}¢>°>,T>l¿>
u>KÓd>U.U>hF>_ò8>À;,>É` >qS>p>5m>Ó÷ð=ÑNà=uÊÐ=WÂ=Tâ´=éZ¨=¶°=©Ô=¬¸=3|=Tk=ØËZ=bK=	~==>X0=¶$=S¸==üA=(ö<*å<·+Õ<®_Æ<~¸<îÉ«<	Ý<Ä<?jL?õ{?[¾Äè¾úö>1ü=Û\z¿øGÇ¼ª4?Ôåc?VîP?shÀ>7{¿Pâp¿NAà> ç"?çÖ¿.p\?ÿý)¿ã?µÃ8¿cBk?­<}¿ÎE#?<@S>|õs¿Wº ?!$?DÝZ¿_"¿f@I?@§E?DR½¾ô¿}fô¾Ý´ý>N~?_>7?Æ8¬¼4¿¯4¿¦ýU¿êå»¾dG>z4*?ÿp?Eg~?[?¢4?é>¢Ä¬½EÊ¾ïÙ%¿	U¿çGr¿â¿/ }¿dëp¿°[¿ì~@¿å!¿4 ¿ÚH¾¾ªªx¾'ó½í(õºá\Û=]bT>V>QYÃ>Úfé>ß´?(Ö?í>"?K.?{8?å±A?¼I?ÝÆP?WîV?ÅM\?9ý`?5e?àh?/·k?gn?Ú¼p?üÃr?­t?Ïv?)aw?x?Äy?dz?î#{?9Ê{?WZ|?9×|?pC}?1¡}?kò}?È8~?¼u~?ª~?EØ~?âÿ~?1"?è??¥Y?ïo?=?õ?o¢?ù®?Õ¹?<Ã?aË?nÒ?Ø?ÔÝ?hâ?`æ?Ïé?Èì?\ï?ñ?ó?1õ?
|±>²(f?ÉÚ>¸Üq¿Å2x?T_.¿®Ù¾íO?Ìy?¨åV?)d?úh?\Ã	?2,¿ºë.¿y|p?ÌZ°¾Vr¾O
?	6¿ÂÔ ?Ôý;¾q¸°¾Õ`?e3m¿{^>á]?¼}2¿· ¿AÄG?[4?N¿%v¿Eiä½!U?±Ùg?«+9>*%¿ÒÀ¿@¿mØ	¾´ ??Hi?D}?z:F?ã­Á>tê½ÉBø¾ª°I¿ÖJu¿H¿¶m¿ãF¿«¿hF¯¾ÆÚ½(ò=õ"¥>¾ ?ý%?/mD?C»[?l?§w?¿Ý}?èû?RÁ~?]Úz?[Þt?YOm?d?Ç[?Q?ÞF?£<?JG2?³3(?nd?+é?ÞÌ?±?õ>êÕå>MîÖ>ãÛÈ>»»>a!¯>Mk£>Co>$>m>«ÿv>ú'f>xlV>Y¼G>j:>#>->³Q!>4>·×>30>!cò=8¡á=Ò=|Ã=óµ=ZY©==0±=õ=`~=~l=õ\=ÒL=Þ>=3c1=*%==Fö=B
=Ò÷<mæ<nÖ<Ç<²¹<Î¬<*Ï <W¥<m p?+à>g¿0Ë§>ßz¾Dn;?ñµg¿Y¿s`>?ÓÖæ>Ãô½~ÅW¿@n=¿Dë:?Â¯>­Up¿H·x?lW¿
J?y8]¿"¦{?zDp¿U³õ>À>Bh}¿eC ?@7?ãCG¿o ¿C®5?uöU?6â¾"g~¿«2¿Ù>Ç{?1CC?ËÏ3=:J)¿~«}¿-]¿ùÛÒ¾UQ>þ!?úl?©F?[ä_?O©?a>§i½ë¾¾@.!¿ãÙQ¿4p¿~¿D4~¿§Qr¿ó¯]¿ôäB¿-,$¿ÒZ¿ÁÂÃ¾Ñ©¾Ðï¾*7¼Í¸É=NL>BV>¤
À>pæ>b?µ©?5!?È,-?%¸7?Àü@?ØI?<P?
uV?îã[?÷ `?ÞÁd?÷Yh?czk?E2n?òp? r?dt?Êïu?Gw?ìpx?.sy?Sz?6{?w½{?IO|?¥Í|?#;}? }?0ì}?b3~?q~?|¦~?ÃÔ~?Øü~??¡=?«W?9n?Â?¬?S¡?®? ¹?Â?ÁÊ?äÑ?Ø?lÝ?â?æ?é?ì?)ï?kñ?_ó?õ?Üz?ã`c?!Ó¾ªîÕ¾à
?0ÐR½ñÌb¿¸©>#1q?ú©?Þó?Ø^?a> g¿EÁ¾¨ü?¿Êa=ÓË>,hÊ¾9>F=I¿×s?\mX¿·¼¼+n?4t¿:È9¿@j2?H?ìé¾¯ë{¿\EP¾H?ÖÕo?$>¿ |~¿2èI¿6¨B¾Çê>¦ßc??~?áL?ÍÔ>Ý½cçè¾ºD¿Ís¿ã¿p¿¢§J¿î7¿»¹¾mÒ ¾ÎôÍ=->ç1ù>Á-#?G8B?§Z?(gk?ñv?}?Aò?Æö~?¥A{?ïku?gùm?gYe?è[?¿òQ?IµG?ó`=?ó3?¨)?(.?«?:?È?Áçö>mç>GØ>¥ùÉ>¾¦¼>5°>X¤>¶M>ëô>ZE>+lx>|g>ªW>8åH>f;>{@.>B">>ù¨>-ó>hÎó=óâ=Å@Ó=¢Ä=á·=ÈWª=y=´==S==Ìám=b]=¿N=³»?=(n2=&=É=yÍ=Ò=ù<ùÇç<b±×<¸È<«Éº<MÒ­<JÁ¡<ª<D\>­>ë¾¡8i¿êh?§W¿$©?Bwí¾­q¿«¾£ËQ=Ó¼¡ßþ¾´A}¿[Ú¾gm?½%<ýþH¿Z?fÌt¿%k?uKu¿qÕ?3Y¿é>»?î¿«¹>KO?· 0¿27¿°?¿Íc?6¾)¦z¿3#¿®³>w?°TN?ö|Þ=,b¿ U{¿Çc¿Ré¾bÖ=Ó}?Ïþh?éÎ?ëúc? õ#?^æ >>2ó¼ùÞ±¾]k¿N¿®n¿V÷}¿Ä³~¿î¦s¿è¡_¿¾@E¿ûÆ&¿A¿ù5É¾xú¾«¾Æ§¼Ú¸=k6D>©>¹¼>µvã>ñ?|?* ?8A,?Øè6?»F@?T~H?r°O?ûU?y[?7D`?qd?ªh?@=k?"ým?É`p?
tr?HAt?Ñu?ì,w?4Zx?z_y?Bz?h{?¢°{?*D|?Ä|?É2}?Ä}?ëå}?õ-~?\l~?j¢~?<Ñ~?Êù~?ê?V;?¯U?l?E?b?5 ?­?)¸?ÊÁ? Ê?YÑ?×?Ý?´á?Ãå?Hé?Sì?öî??ñ?9ó?ðô?¯j5?¥&>v¿/m»>èÍ3¾E.?&o|¿ÂÌ¾]M?¤ d?å_?Ó?V¾õ³¿¢j³¼Ë5r?¶0U¿~I¯>eý¼<0£¾ãLY=¬ÿ>»ð0¿´5~?ÞÓ<¿>ç?¾Öz?+ªî¾ÙO¿ÍQ?®'Y?Xa·¾Ø4¿[D¾5:?}Cv?PÉ¦>+È¿Í$|¿S¿tÚz¾<¨Ò>½ì]?i?+S?>æ>j»$GÙ¾vR?¿äp¿Óÿ¿J>r¿NN¿X½¿áÂ¾Á5¾ê°©=b>2Gò>ÝU ?ú??rgX?»9j?J4v?á'}?íâ?U'?À¤{?ûõu?} n?;f?E²\?ÆÄR?%H?7>?«ï3?×Ò)?B÷?{m?+A?
z?a9ø>Vè>åKÙ>Ë>³½>Ù±>½E¥>
,>%Å>1>Øy>Ñh>|èX>J>U1<>ÇB/>l3#>õ>4z>"¶>¨9õ=ôEä=å{Ô=ÇÅ="¸=4V«=Ww=8j= =Û=Eo=*­^=ë:O=Ú@=y3='=n=ª¤=Í=Wú<ß"é<8ôØ<üäÉ<Aá»<|Ö®<k³¢<ýg<þ4¿|¿¶c¾
;n?Ê|¿ø\L?Q*>×÷v¿0I¿úÐè¾@@ø¾ÇS¿^v¿øBE½Hð?%Æ¥¾¹¿¤p?î¿k}?¶£¿Zw?'9¿4Çñ=ÕÛ,?àv{¿ýD\>Y|b?îÜ¿%BL¿`?>o?	"¡½î¹t¿lâ/¿>Ü>L	r?eX?ö1>ç¿3x¿îSi¿:ÿ¾6t=Ä·?«§d?Øÿ?Îg?ù*?°!¯>4»v¥¾ö¿|-K¿»l¿ìM}¿§¿!ët¿Ja¿)G¿2Z)¿hÄ¿S¢Î¾!G¾ä7¾Úûó¼_e¦=|<>´ù>$f¹>ðzà>"º?;M??T+?6?Ö??ÞG?,$O?U?[?÷æ_?Þd?úÌg?Æÿj?³Çm?\2p?ºKr?It?:³u?w?ZCx?ªKy?×0z?÷z?º£{?û8|?Qº|?d*}?~}?ß}?(~?¡g~?Q~?¯Í~?·ö~?@?9?°S?Æj?Å~???¬?Q·?Á?~É?ÌÐ? ×?Ü?Yá?uå?é?ì?Ãî?ñ?ó?Îô?·ÞW¾^Ë1¿CU¿}Âl?©L¿?ty?KN/¿cÉC¿$DÕ=õP?·È?È[>N$¿!ûp¿ÆR¬>@»H?±v¿è£?ý~¾Ó=hj;¾d>ë>¼òR¿R?ÁE¿¯|±¾Æ~?i«¾¢a¿³ÿ>[Tg?+¾8ñ¿%Ã¾*?ø{?~Ê>¸ò¾_½x¿¿h[¿Ö ¾¼õ¹>"sW?Cù?GY?¥ø>=«fÉ¾9Ù9¿¨àm¿eë¿0Ft¿½×Q¿X/ ¿bÌ¾9'¾J_=%â>ÈMë>u?	´=?¶°V?Oi?³ou?9Ã|?íÍ?þR?¬|?}|v?Do?Îf?ìy]?S?_I???qÂ4?A¡*?º¿ ?Þ.?²ú?5+?sù>é>&zÚ>J4Ì>À¾>L²>¹2¦>>
>D>ôÊ>ÆD{>d%j>]&Z>À6K>5F=>E0>;$$>Õ>jK>y>ß¤ö=Hå= ·Õ=÷ìÆ=`'¹=T¬=3d =¹F=Èí=ïK=<©p=Cø_=oP=ZùA=4=ý'==U=Ü{=c=ûû<Å}ê<7Ú<kË<×ø¼<«Ú¯<¥£<OI<5?z¿08¿8?9½Â>È¿jf>Ö:?aï$¿³~¿ÅV¿¿wV¿%_}¿ÙPD¿tÊ¬>q?Ná¿!Ì¾ÅL?%þw¿£~?­{¿ö`c?y
¿nã½ðK?N p¿-=Í4q?×ßñ¾åÉ]¿µKÛ>Ôxw?ý­<¯l¿4P?¿ÚG>½k?Ëga?]$r>xä¿QHt¿n¿D
¿føj<Õ¯?gö_?dÙ?¿\k?È0?06½>¾hÍ<m.¾½¢¿±G¿!®j¿Õ|¿ât¿(v¿û\c¿ÙI¿´å+¿*q¿Ô¾¥¾6V"¾Ü ½±¶=mý3>UH>g¶>¾|Ý>d ?9?Ë?÷f*?]G5?Ø>?î<G?2N?WU?	£Z?7_?6Îc?èg?õÁj?÷m?®p?0#r?ûs?°u?øv?`,x?½7y?z?èz?¿{?»-|?°|?ñ!}?,}?GÙ}?#~?ßb~?2~?Ê~?ó~??¶6?®Q?	i?C}?È?ô?«?x¶?SÀ?ÜÈ?@Ð?¦Ö?1Ü?ýà?%å?¿è?Ýë?î?æð?ìò?­ô?ù»o¿íë}¿oò½i3j?¤~¿d?Ö½Ó~¿Odâ¾yU=Ôf¹=¾ÒCf¿=¿?&?}K?f»¿mN?{9 ¿þo³>QUÑ¾-$?sl¿Ù#w?è£é¾U,þ¾½ï~?ìÀI¾µo¿¾ÔÆ>2lr?åk¾í~¿¾=î¾yë?AK~?+>í>k²Ò¾It¿Zùb¿X´¾åÀ >ÂvP?à?<^?·±?Ý÷=©J¹¾i04¿±ýj¿Ö¥¿{%v¿¼BU¿`$¿'Ö¾	Ñ:¾»B=¸>Fä>ó?éd;?ñT?ïÆg?Û£t?X|?B³?Ày?g^|?rÿv?¼åo?og??^?´cT?¦2J?øß??C5?ãn+?!?½ï?Í³?Ü?øÚú>KÕê>¨Û>,QÍ>YÌ¿>³>§>Qè>Je>¢>à°|>§yk>(d[>j_L>[>>>G1>%>üµ>>ý;>ø=êæ=òÖ=eÈ=8º=S­=Q¡=9#=»==òr=ZCa=A£Q=,C=5=÷õ(=v<=S=V+=Úoý<«Øë<áyÛ<Ù=Ì<m¾<ÚÞ°<¬¤<¢*<³¾a8>33~?ù¾Î¾m¨×=pmè¾~?ÜÃ½=e¿ï¦¿çò~¿ZÑs¿»ß¾;ü+?^fB?®³X¿ð];=¯g?]¿wÃo?$ i¿xD?õÅ¾¾­Êc?4:^¿u|º½eûz?Wº³¾oèk¿Ý¤>}?G±÷=áb¿òNM¿lì=àÛb?¢Qi?:>LÍì¾Õo¿ s¿â¿²ý¼IÔü>ÛìZ?[?Ó¥n?Ø5?¾ Ë>OÈV=í®¾i¿ÉD¿Ðh¿"´{¿o¶¿î?w¿ß%e¿gL¿bi.¿m¿®eÙ¾ÛÓ¾ap,¾?*F½)=`Ü+>ù>e¸²>)|Ú>Èþ>ì?y?Hx)?1u4?o>?F?	N?T?ó6Z?ú*_?|c?t>g?Ïj?ð[m?½Ôo?lúq?µ×s?úuu?qÝv?Dx?³#y?1z?{Ùz?±{?k"|?Ä¦|?s}?Ð|}?çÒ}?{~?^~?~?Æ~?ð~?á?a4?©O?Ig?¿{?x?Ñ?ª?µ?¿?8È?²Ï?+Ö?ÆÛ?¡à?Öä?zè?¡ë?\î?¹ð?Åò?ô?(M¿;(û¾°þ-?r8¯>ê'¿`6È>8?¦_¿0Z¿hè¾§¤Å¾æ0¿qá¿XàÚ¾·a?a>ího¿íwq?þ]7¿ØB?ý¿BzK?fP{¿Áõe?Þ]¾õ!¿Xx?Û@a½my¿ç>]Iz?Õ3½ Ây¿I©¿?Ø?½H?¬¥±¾În¿×µi¿<þÎ¾d>ÛûH?aA?Þ½c?O.?~tÛ=â÷¨¾yY.¿£æg¿5/¿ÛÛw¿X¿äÖ(¿¶¨ß¾aN¾]zò<7	y>0Ý>;?A9?ê)S?¥f?ÈÐs?æ{?ë??ð´|?Ø~w?àp?Ê9h?_?0U?GK?¨²@?"e6?¾;,?ÃN"?°?|l??ï*ü>ì>ÕÜ>ÂmÎ>iØÀ>´>;¨>EÆ>55>;P>Ù~>ÎÍl>Ý¡\>M>Ëo?>gI2>½&>h>Ãí>äþ>6{ù=Ý<è=&-Ø=Î7É=ÒI»=hQ®=å=¢=·ÿ=N=Ê=¦ps=ob=j×R=ý6D=ô6=iî)=¯#==*=ó=äþ<3í<¶¼Ü<GjÍ<(¿<	ã±<Í¥<ô<C6??_?þÇ;?¿p¿e<A?k¿MüZ?[Hù>:ç¿hd¿H(l¿Ø9¿#ú¼Æmg?O2ô>=½y¿Qµ>ÿª>¤2¿gQ?¿J¿ýX?#C¾Ðûà¾2Ýt?ÃAF¿x¾Ó?¹Çd¾*fv¿ö!W>ÎÀ?ß´`>8V¿ËÃY¿ç	=UY?ºp?{¸>XôÐ¾p%j¿ív¿¿½ÒÕé>U?©~?]¨q?¶s;?FÞØ>Y£=
¾±¿n@¿dHf¿çÃz¿Iã¿_Px¿Ùàf¿ýFN¿å0¿º¿Q¼Þ¾¾"6¾<:l½9¢b=w¸#>­ß>)^¯>7y×>öhû>¸¹?ó?(?¢3?îe=?cøE?{M?T?NÊY?=Ì^?)c?öf?QEj?%m?¥o?nÑq?´s?Wu?¥Âv?þw?y?·üy?UÊz?|{?
|?è|?è}?iu}?}Ì}?í~?FY~?â~?êÂ~?cí~?,?2?¡M?e?9z?&?¬?©?Ã´?Ø¾?Ç?#Ï?°Õ?\Û?Eà?ä?4è?eë?(î?ð?ò?iô?æ=ãÅÏ>ãÄ?J¸á¾==óa¾xn?u`â¾[û¿¥V¿ó×E¿p¿÷©l¿lØI½ÚÅ}?Ûñ½wXG¿Ã¿?LØ`¿ÑJC?(ÏG¿ýh?Ôù¿lhL?Iñ½?g@¿Õßk?´=ë~¿9­>Ñ~?Ò=yçr¿Pþ¿Õæ>¶»?º/?»¾ÔPh¿èo¿nüè¾b.Z>÷A?~?×wh??s?Ó>)s¾èU(¿-d¿~¿iy¿(¼[¿]-¿­é¾w&a¾B«A<êh>Ö>£?,­6?ÿYQ?}4e?ör?wn{?êl?¸?F}?¯úw?q?£ëh?pÄ_?ÁûU?{ÔK?+A?57?Ð-?R#?ío?¾$?§<?Vzý>JRí>»Þ>Ï>;äÁ>|µ>¿ø¨>¤>>¿>°>Ù!n>|ß]>°N>@>K3>oö&>Ìv>æ¾>ÅÁ>Væú=é=2hÙ=3]Ê=[¼=ÉO¯=»*£=3Ü=U=#=WÔt=Ùc=T=ÍUE=ä¤7=Ùæ*=ç
=m=ß»	=®, =uî<ÿÝ<µÎ<?À<7ç²<í{¦<Gí<Ã[?7ùi?1ï-½Çe¿Ù?Üt¿w¼>0e?
C<_¿Lv"¿}×±¾Q4Ã>a°?cÇ>­6~¿ ?HP5=¸Êô¾+%?Ó	 ¿xÖ>ÍÜ`<¿§6~?¤Þ(¿ýÆ¾Ë ?¼½È}¿,±Ä=¾{?*©¡>G¤H¿Üd¿³ð:½N?¡´u?Ê×>X´¾õc¿!z¿ú'(¿Roø½ìlÖ>ÿØO?ÕZ}?xct?ëÞ@?ºkæ>/Û=¸×f¾SY¿P¨<¿ðc¿;¼y¿lû¿hOy¿Ðh¿»mP¿ÏX3¿V¿Zä¾¿N¡¾7@¾"½Ç5?=Ö>ýP~>¾¬>òsÔ>Ëµø>=?â?Õ'?Î2?«<?òTE?ìL?ùS?]Y?m^?Öb?c®f?}j?ÿîl?vo?6¨q?Vs?
8u?´§v?§æw?Gûx?%ëy?»z?\o{?|?þ|?P}?øm}?
Æ}?W~?oT~?±~?I¿~?>ê~?s?¬/?K?Âc?±x?Ò??¨?æ³?¾?îÆ?Î?4Õ?ðÚ?èß?5ä?îç?(ë?óí?_ð?wò?Gô?¹©`?q²y?o?ìs¿éÕ4?V¿Av?ÑI>*ðV¿Ö£¿Þz¿ÝÂ~¿Tû/¿/@¬>y?Teâ¾%¿Å
x?y¿Àµf?VÐg¿sVz?áy¿j+?t=¦Y¿uÒY?xk>ãÇ¿IyÙ<Ïó?*A.>i¿!ó0¿U¾>}ö}?,=&?§<Z¾ìØ`¿t¿¿%>ë8?q|? Èl?G}?Ù¹7>_Á¾@'"¿
a¿¯}¿½Ìz¿¦É^¿C*1¿zaò¾2t¾Ä[Ã»PX>ÝÎ>
£?ÆD4?ÓO?ßc?	r?þïz??A?Ð?gU}?ôrx?'·q?ùi?Ä`?/ÅV?A£L?TB?ý8?Ó-?<Û#?</?Ü?rì?.Éþ>î>/ß>
¦Ð>ÏïÂ>(¶>å©>Ê>½Ô>.Õ>2z>Éuo>_>üØO>&A>M4>ç'>(W>>¡>nQü=Yáê=8£Ú=Ë=7l½='N°=¤=­¸=Î"=2H=8v=$e=¹?U=tF=Õ¯8=Jß+=ò=Ø="
=ç =Zéï<]Bß<#ÃÏ<,WÁ<fë³<n§<Î<uõ>öÓa>+ÞI¿°h¾ñ35?ü ¿<Â¾ÊC}?G?a&Y½?üK¾ø;É=¥ë9?ßq?
l¾üe¿¥V?~U}¾Àd¾äÝ>=Ù¾Ú-V>^>×#>¿?aÞ¿R¿*y?Zv)=çè¿5¼"D|?dZÑ>Üÿ8¿]³m¿,í ¾I®B?Yz?kÈô>W¾']¿Û¡|¿XZ1¿"*¾l¢Â>ÓI?Ø{?UÖv?ÁF?Æó>·m	>üXM¾4ü¾Ë8¿:a¿7x¿Öþ¿ö<z¿¨,j¿R¿UÄ5¿1ì¿Ré¾¦¾^£J¾Õ$½©Å=h>öÞv>-£¨>blÑ>L ö>Q?îÐ?¦&? ù1?Xð;?¹°D?>\L?CS?YïX?I^?6b?Èef?SÇi?¸l?]Fo?Å~q?[ls?Ïu?v?'Ïw?ææx?yÙy?È«z?b{? |?|?¬ÿ|?{f}?¿}?¸~?O~?y~?¢»~?ç~?µ?M-?I?úa?&w?}?^?§?³?Y½?GÆ?Î?·Ô?Ú?ß?äã?¨ç?ëê?¿í?1ð?Pò?%ô?Ç¨`?opB?ýO¾bù`¿NÔ?Y!~¿0¹?ý.?e×¾ 6d¿!u¿ýZ¿¥î¨¾ÐÅ+?[S?FB7¿ó¯¾[?À°¿:{?ô×z¿Àü?:Ii¿þ-?	Þp>"Öl¿[B?¿y»>|¿ÜsÇ½Ã­}?ã¼>ö ^¿ó_A¿E>Mz?i[4??ò¾`nX¿·x¿XW¿qïà=ÒÂ/?Az?a¬p?<I%?áa\>éÎm¾Ï¿ p]¿ð¥|¿Ã|¿·a¿35¿û¾Û¾Â¼pG>¡Ç>á?,Ô1?y¡M?¿b?j,q?kz?ì?¸ã?R}?¥çx?GLr?ÉGj?ÿ@a?áW?pM?«#C?øÑ8?.? $?î?û?ä?» ?vÍï>ñ[à>»ÁÑ>$ûÃ>¢·>OÑª>[_>Y¤>>û/>Ép>vZ`>^Q>¾­B>O5>¸×(>}7>a>yG>~¼ý=3ì=9ÞÛ=ð§Ì=d}¾=L±=`¥=&=ð=@=µw=§of=ÞsV=jG=Äº9=º×,=VÙ =Ì¯=eL=m¡=>Dñ<1à<ïÐ<ÁnÂ<ï´<-`¨<ë¯<ÿxõ¾&¿;¶z¿Qô>î=ö÷=ßóK¿Ô4;?v>h?þþç>£¦>Ø&?ú©q?éÐ=?Û2¿rÀ2¿8cw?Ø¿³_I=ÒD>k}L¾G'#¼¡×Ò>{<[¿+Ñx?]Â¾ýP&¿8n?h2>xÈ~¿	¾|#v?ôþ>Á'¿Öu¿öGR¾ß­5?dP}?r¹?Ár¾ÜqU¿s~¿!:¿ø³W¾T®>}C?> z?< y?âK?9u ?K)%>¸3¾:ñ¾ÈÖ4¿ ö^¿õfw¿í¿ú{¿I½k¿<T¿'8¿m|¿âî¾¶«¾TªT¾µ#¯½¥ð<ô<>bio>B¥>bÎ>Hó>á
?@¾?Q³%?#1?D4;?ºD?ÁËK?îR?X?­]?`/b?Êf?Ói?ßl?co?Uq?-Hs?iùt?^qv?·w?hÒx?µÇy?az?»T{?ô{?þ~|?üö|?ô^}?¹}?~?¬J~?<~?ö·~?æã~?ô	?ê*?vG?0`?u?%?5?¦?)²?¼? Å?sÍ?9Ô?Ú?,ß?ã?bç?®ê?í?ð?(ò?ô?ÉÖ= ½Ì`¿mV¾A?3á.¿BÉ¹z?GÙ>Á³¿.7¿ÄK	¿"Ê®=XNg??>\h¿º¬e=)&+?pÂr¿n?TÕ¿À8y?JãN¿"D°>âÎ>y¿"Í&?`ý>;òs¿ba¾ªx?¹À>+P¿e P¿ç¨R>Ôu?"vA?ôÞ½BO¿¾ì{¿E¿1Nl=~&?_w?7#t?	Ô,?W_>ÉÔK¾	O¿àY¿Dl{¿ã}¿ÿd¿M%9¿ÌX¿)¾7*½ØÊ6>_XÀ>9?y[/?¹K?@a?©<p?Éßy?ñØ~?ïñ?å}?ÁXy?aÞr?òj? üa?ÓRX?<N?¥ñC?ù9?Ig/? e%?E¬?õJ?üJ	?² ?\
ñ>ýá>ÝÒ>;Å>éý·>Z½«>Ë< >Ûs>ÊY>²å>Ur>Ña>®)R>GÂC>Q6>NÈ)>Ê>*2>K
	>'ÿ=»í=5Ý=HÍÍ=¿=ÜJ²=/ñ¥=q=H½=LÆ=aÿx=¶ºg=¨W=7²H=³Å:=)Ð-=À!=û=¨=Ì[=!ò<Èá<ýÒ<VÃ<Âóµ<LR©<=<ç[¿h¿Õö¾åv?'¿õ:?ÿÿ¿Z\[>Áæ}?gV?Õ2?«X?â?eÛ>|:R¿säÖ¾ä?ê`>¿¢>Ii{½TÉ=(j¾¬Æ?ÖYp¿Ì=j?ì]c¾þ2B¿rd^? B>ê¼y¿Ã}¾ä+m?Ú ?Á¿6z¿®"¾	¨'?ÄF?!{?/ü5¾Ò)M¿Ù¿ÑvB¿j¾Ð> Û<?¢Òw?àz?óO?ëê?+Å@>_ù¾Ëæ¾ôË0¿µT\¿v¿Ç¿dã{¿?m¿ÇV¿r:¿§¿Éó¾ïâ°¾Ù«^¾´Â½Fº©<ø>\ðg>Èß¡>VË>pð>å?ª?¿$?0L0?Vw:?õeC?:K?ùR?*X?\L]?Ûa?kÓe?ýGi?^Il?'æn?4+q?Í#s?×Ùt?ùUv?Ãw?Ì½x?×µy?ãz?NG{?âè{?èt|??î|?bW}?y²}?c~?ÀE~?ø~?D´~?³à~?/?(?bE?c^?
t?Ë?	?¥?H±?Õ»?øÄ?áÌ?»Ó?ªÙ?ÍÞ?@ã?ç?qê?Uí?Õï? ò?àó?WM¿{7W¿Æ(o¿@?>ÊÛ=èå]½Cá¿îl?	#?¯Ó\½g¾H¼æ½½«ø>¸¬?u>ìâ~¿FBº>È%Ù>Ú¼S¿Ns?ùv¿s^f?¾Ë+¿ï#>x?0¿O
?B?ug¿I­¾Ëo?þ|ò>³@@¿É]¿Ïfö=uÜn?zzM?Ã±»¦æD¿k6~¿Eg$¿¿°;7Ô?Yt?ì*w?·4?ìa>h)¾Å¨¿U¿Tz¿ñü}¿0g¿r =¿÷×¿Ic¾ãÊr½M&>
¹>;t?ÊÚ,?ÈI?²_?ÏEo?Ny?P~?<û?&~?FÆy?tms?Ðk?#µb?Y?ôO?n¾D?k:?/00?)&?ýi??ºù	?'Y?ËFò>©³â>4øÓ>Æ>þø¸>;©¬>¡>BC>ø>X>ðps>Õb>ëQS>ÁÖD>S7>Û¸*>ø>5>Í	>BI >â×î=,TÞ=òÎ=µÀ=2I³=ûÝ¦=N==X=cz=Äi=&ÜX=ÑI= Ð;=È.=Â§"=)^=ëÜ=+=úó<Ö
ã<jHÓ<êÄ<ð÷¶<lDª<r<®4¿Ï 
¿s¢¶>îÊ[?~¿Ã?rÕK¿yæÁ¾\E?° ?VÃs?¾^~?;Ç_?ÐmN=µx¿²Ð¾½un?èÕg¿Èâ?Ã;¾X(>ÈMß¾Ë=?·|¿"GT?)o½e|Y¿4J?¿Ú>Úp¿ ·¾Õwa?
)?­¿ò#~¿<0¸¾õ°?
ÿ?l#?ñ½¼;D¿ÿ¿VJ¿µ¾,T>î5?mPu?«v|?ëT?´B?
<\>! ¾¼äÛ¾«,¿ Y¿.µt¿Ä¿%|¿³n¿
X¿ÒÔ<¿Ä¿×÷ø¾
¶¾ª§h¾dÕ½]E<½õ=ÿs`>	{>BHÈ>Òí>­?°?ÇÊ#?dt/?¹9?k¿B?²¨J?fQ?¿¢W?*ë\?fa?©e?Ñi?l?ªµn?q?;ÿr?ºt?n:v?Þw?©x?ß£y?P}z?Î9{?.Ý{?Åj|?vå|?ÅO}?á«}?­û}?Í@~?¯|~?°~?{Ý~?e?&?KC?\?yr?p?Ü?¤?g°?»?NÄ?NÌ?<Ó?<Ù?nÞ?îâ?Óæ?3ê?í?§ï?Øñ?¾ó?G»o¿Y½o¿öÉª¾Mty?(^¿X ?Æv¿ßâ?*s?íç>ºÓ3>OÞª>	M?,q?jû½ÈBx¿+«"?¨>Ñï$¿®`W?`_¿ëWH?|¿²û¼±a2?È~¿¡*È>¦~8?¯ÜV¿?Ìç¾±íb?%º?j.¿mh¿*F=7¬f?(WX?q¹=Þ9¿¿7,/¿é-@½2Ë?l£p?!Ây?g;?2¤>ã5¾ûÝ¿Ø?Q¿fhx¿Ç¸~¿Èºi¿Ä@¿H¿Z¶¾óº½Z>z¤±>V?=R*?ÐG?B>^?âGn?þµx?
Z~?ÿ?Ãc~?20z?ùs??l?lc?uÙY?õÏO?E?6;?Hø0?iì&?-'?·?¨
?pÿ?Ãó>õÞã>ûÕ>­Ç>ßó¹>ô­>H÷¡>>Þ>ìP>oÄt>Cd>zT>,ëE>jU8>]©+>MØ>9Ô>á
>¾þ >*ð=ß=ìÐ=×°Á=G´=ÆÊ§=*=½W=bD=´Æ{=ÑPj=GZ=ÎïJ=Û<=Á/=ø#=W5=-¥=Ð=çTõ<¨Mä<×tÔ<~µÅ<ü·<6«<àS<¶³>ª³>fVq?f>UùL¿¾ÇL?Oa¾±fV¿¢ >ÌPd?|?ÌRq?bH?Üµ«¾N~¿¦áy>|¬E?d\}¿ÉC?Ä`
¿ëú>}^¿°Ö\?á¿7?¾ÖÚ= k¿Õy1?_-?Cd¿üùì¾á)S?Ëc;?Qó×¾Ú¿ÂÞ¾,Þ?Mx?)	0?¨l½Õ®:¿Ó·¿r»Q¿¹°®¾³½`>M¹.?|zr?.Â}?høX?.{?£w>ÉfÌ½9ÞÐ¾±t(¿ËV¿ê9s¿X=¿1C}¿óp¿éZ¿?¿¬ ¿(þ¾Ö+»¾r¾\è½á^;6Yå=fôX>P>Ú7Å>ë>ús?Ñ?Õ"?«.?îú8?B? J?4Q?Æ2W?z\?B1a??e?OÇh?yÙk?ên?½Öp?vÚr?/t?½v?Ùow??x?Ïy?§mz?;,{?jÑ{?`|? Ü|?H}??¥}?îõ}?Ó;~?_x~?Ò¬~??Ú~??­#?1A?ÂZ?æp??®?£?¯?Mº?¤Ã?»Ë?¼Ò?ÍØ?Þ?â?æ?õé?éì?xï?¯ñ?ó?þÖW¾¼l¾ ?l>V?Oî{¿Ly?îóm¿hÖ¼ex?;eV?ñ©?âv6?¿Ðy?&>?Þ*ð¾Í=U¿wX?S¾Y°Ó¾6-?w¿;¿ô ?w£¾\Ja¾£P?v¿Ã{>ÿòO?(sB¿ËB¿ÁÃS?Z&?SÖ¿à$q¿~LW½´ü\?üa?4ô>Ý.¿Æÿ¿dc9¿öÊ½i?@ol?§ç{?WÐA?Ëµ>ÎJÉ½kð ¿ÇÑL¿Év¿EJ¿ä#l¿oD¿'ª¿|ú¨¾±Â½v>!:ª>ã0?ðÁ'?ÂÏE?ÛÂ\?éBm?x? ~?ÿ?É~?z?~t?¬ál?Ì d?"Z?P?jTF?  <?¿1?¯'?Ôã?Gm?&V?n¥?C¾ô>à	å>t-Ö>&È>îº>®>TÔ¢>¾á> >o>Òv>ZOe>.¢U>ÿF>@W9>Ö,>¸ >7¥>¤R>5´>|ñ=	Êà=7=Ñ=÷ÁÂ=ÖEµ=·¨=÷=õ$=l=[*}=Ük=hD[=L=yæ==r¹0=-v$==om=è=É¯ö<zå<C¡Õ<ÍÆ<K ¹<«(¬<25 <?z?Zåu?"i]? ¿xÙ5¾tÉh>'Õ¼>é¿å½w¾å?,°I?
3?-¹_>W+¿b¿×¥?ïH	?}¿%i?§3<¿ã.?b[G¿'r?5ºy¿S+?æÑ><0x¿O?R&?;)T¿Ø¿FlB?ÁÝK?åà«¾g¥¿,;¿áð>8³}?í¹;?´+;×0¿a½~¿¡X¿ PÄ¾Àl6>¹?'?ÆQo?®Â~?[(]?ö?ÛR>i½¤¹Å¾M)$¿KãS¿é§q¿CÙ~¿{Ø}¿Ìoq¿Ll\¿­_A¿D"¿æ¿HÀ¾.|¾.ôú½RQ¬»þðÔ=®qQ>©«>P%Â>ØRè>Ò9?æh?CÞ!?Â-?w;8?
pA?ÞI?cP??ÂV?L'\?°Û`?õd?wh?¡k?èSn?+¬p?µr?zt?æv?²Ww?Lx?¥y?è]z?{?Å{?RV|?¾Ó|?k@}?}?'ð}?Ò6~?	t~?©~?ÿÖ~?Çþ~?<!???íX?Po?³?}?¢?®?¹?ùÂ?'Ë?<Ò?^Ø?®Ý?Gâ?Cæ?¶é?³ì?Iï?ñ?wó?l5?¨?_é{?±=>®VW¿ojd?,m¿{ì¿N1?|?H·i?q©r?ÔÒz?¹éÛ>ç@¿RÕ¿*x? QÜ¾u¾¼üð>4¿ö#â>x]ô½WuÍ¾Å'g?âÔg¿°¿=$c?º*¿lk(¿³ÇA?´n:?¶¿x¿¾ÞQ?¾\j?aO>X!¿È}¿C¿ç£¾Xhû>Í¾g?}?á9H?%Ç>Uî½±Ãó¾P6H¿Õ¥t¿R±¿kn¿§H¿8ü¿Ó.²¾¹<æ½ç=sÅ¢>½	þ>ÿ)%?¤ÇC?ë?[?í6l?»rw?Ä}?§ù?Ñ~?;ùz?qu?Äm?nÓd?
Y[?]Q?G?4É<?2?q(?ñ?"?Ô?#K?Lùõ>i4æ>G×> 0É>é»>çk¯>>±£>Ô°>ÿa>à»>kw>Yf>3ÊV>ÖH>Y:>F->°!>.v>b>©i>/Îò=ðâ=~bÒ=ÓÃ=$D¶=T¤©=fã=+ò=tÂ=ÿ~=åæl=x\=b-M=eñ>=Þ±1=b]%=²ã=±5=GE=«
ø<LÓæ<¯ÍÖ<¦äÇ<xº<Ê­<¡<4?s L?¸F6>×{¿Sp
?¸0ç¾X[?¶ÇN¿Õ8¿ñ`=äíÐ>"£>éáL¾Ø.g¿
L(¿õL?S }>:g¿a}?½Þa¿3DU?­e¿Ï+~?ã{j¿tÜ>&*Ù>¦à~¿m)ì>`â>?"Ì@¿¼I'¿vp/?ANZ?m£|¾=}¿¿¿IÎ>þ±z? F?H
=ø×%¿d}¿0_¿WÙ¾ÅÈ>9?`×k?àw?²a?³?	Ç>á¥H½xº¾rÉ¿©äP¿Pÿo¿`~¿ú[~¿þ·r¿D^¿óC¿tñ$¿L(¿^Å¾/;¾¸í¾d¼<Ä=òëI>A>®¿>ïå>þ?ñP?æ ?zç,?({7?4Ç@?ìîH?õP?-QV?¢Ä[?¯`?ªd?IEh?hhk?¥"n?_p?Vr?ØYt?èæu?j?w?=jx?bmy?Nz?Ü{?±¹{?L|?ÐÊ|?®8}?ß}?Xê}?Ê1~?®o~?J¥~?¹Ó~?òû~?È?ô<?W?¸m?R?K?ÿ ?º­?À¸?MÂ?Ê?»Ñ?î×?MÝ?ôá?ûå?wé?}ì?ï?^ñ?Tó?oz?¨ý?íE?yÄ¿0ã|¾}É>ÈØ=9¯q¿v2M>lkd?n?à}?£âO?#S=Ip¿©¾co?Ìr,¿ø=ï¨n>¼¯¾8p>ée¯=Oe¿øv?/ÇR¿¯åv½ ­q?à°¿õü>¿Ù6-?q,L?cÞ¾Êá|¿^=e¾£bE?¸kq?¶e>F¿~¿ÓL¿+`O¾ðeå>b?òÙ~?TN?(<Ø>s«ù¼'hå¾wnC¿ê}r¿ßí¿Ìp¿Ê|K¿O>¿R»¾1¾èÊÅ=ãF>Y¤÷>"?Î·A?~µY?õ#k?Çv?iq}?Lï?$?UX{?Wu?Ln?íe?+\?4"R?åG?K=?¹K3?d2)?[?M×?&±?ð?Û3÷>^ç>yaØ>ù9Ê>Nã¼>"W°>¤>Î>Õ#>?q>A¾x>AÉg>&òW>(I>ÉZ;>«z.>Õx">G>Ø>>< ô=Ñ?ã=ÁÓ=+äÄ=oB·=ª=Ô¿=a¿=z=¢ñ=í1n=¥¬]=*LN=Oü?=Jª2=D&=ßº=òý=¥ÿ=eù<è<ú×<9üÈ<¥»<é®<Õ÷¡<FL\¾<
¼\"¿;VP¿x?ñWk¿?~?Ð¨¾¶Îz¿Ð,ç¾]½©¾Öe¿ú¨¿e°¾Ü2t?ú½Ï3=¿Z~?2óx¿´p?*Ûx¿6?+´R¿¹Ç>¼I?Õ¿YÝ¨>¼ÞS?öw*¿d<¿n?[f?gU¾®y¿¾#¿BQª>]xv?ï¯P?S>ü=à¿	²z¿öÝd¿8Nî¾TÊÁ=Ø?yh?á?hÝd?[%?E¤>¯®À¼¶¯¾¬U¿{ÏM¿I@n¿>Ó}¿¥Í~¿tñs¿4`¿RÇE¿$\'¿.®¿PoÊ¾k,¾\^¾ñ¸¼6´=OcB>¸Ô>ûù»>áÊâ>GÂ?ó7?Ðí?,?º6?@?JZH?éO?ßU?{a[?@/`?Ô^d?Æh?n/k?ñm?YVp?újr?j9t?ÅÊu? 'w?Ux?[y?(>z?{?¼­{?¦A|?ÕÁ|?æ0}?"}?ä}?»,~?Lk~?~¡~?pÐ~?ù~?Q?Ñ:?<U?l?î??ô?Ó¬?ù·? Á?üÉ?9Ñ?~×?ìÜ?á?²å?8é?Fì?ëî?5ñ?0ó?Ux±>Æ/?Ô<·Q}¿¿õ>¾òr/?p×v¿\¥´¾p?QÃY?*×U? ?{+«¾}ú¿»U=È{m?é[¿'Å> ¼ì¨ä½y<ò>ÚÄ6¿5ÿ~?9÷7¿7ÆY¾©A{?·}ä¾R¿UW?-[?Y¯¾B}¿4Y¾L7?Pw?sn¬>·k¿Ó®{¿ÿeT¿Å¾KØÎ>3ó\?_¥?ÊT?âé>Öae;ãÑÖ¾L{>¿s'p¿àÿ¿ör¿ÝN¿÷o¿­dÄ¾ù9¾ë¤=æ¾>ð1ñ>­ã?T ??¤#X?

j?"v?}?à?u.?Ñ³{?,v?Bºn?G2f?Ñ\?YåR?X¬H?bX>?4?ó)? ?¦?^?¬?ñmø>Uè>{Ù>CË>`Ý½>3B±>­j¥>¬N>å>&>Mz>i>Y>C<J>{\<>k/>óX#>
>Î>Ô>Brõ=­zä=ÿ¬Ô=@õÅ=·@¸=Ù}«=@==@=¡ª=ó|o=Âà^=ñjO=9A=µ¢3=É+'==3Æ=º=mÀú<îXé<&Ù<ÍÊ<Ò¼<ÿ®<&Ù¢<!p¿N¿õ¿õÍ¾0`?)u¿^l:?V¹>\o¿EV¿Ô¿æ½¿4f]¿ Eq¿ÙT<Ü×?n.¿¾5¿kBl?jó¿=f~?{÷¿Ì:u?­@3¿µ=22?Í$z¿6D>e?¡¿A4O¿t¤?´p?ÔL½qs¿á`2¿(ª>q?ÞY?QC;>¤è¿ù£w¿Ò-j¿lK¿ÑUW=»S?\òc?ÿ?`h?Ë+? V±>as ;§£¾Î¿¤J¿ûjl¿d1}¿s-¿u¿Ða¿°íG¿;À)¿x/	¿
zÏ¾*¾ÌË¾~Ûÿ¼1¤£=Þ×:>f>Aá¸>´à>ê?ï?!ô?¦/+?ø5?Bs??ùÄG?@O?bmU?ØýZ?cØ_?,d?îÁg?*öj?X¿m?+p?mEr?Ñt?{®u?vw?È?x?Hy?(.z?0õz?¶¡{?;7|?Í¸|?)}?[}?¢Þ}?¤'~?äf~?­~?!Í~?<ö~?Ö?«8?`S?j?~?â?è?ë«?0·?òÀ?eÉ?·Ð?×?Ü?Ká?hå?ùè?ì?»î?ñ?ó?>¿¸¾S@¿äJ¿¾õr?0V¿w|?$¿ L¿'.d=|w?¸£?É&Õ=ÐX+¿ðém¿Xr»>ûC?Hxx¿ÜË?AÀ¾zÙÿ=3 P¾òÞó>û|U¿û~? ¿ÉU¸¾·¯?
¥¾âc¿Kîú>wih?ªæ|¾¿â¿ÙåÆ¾Ýª(?Qo{?Q Í>«ÿï¾lex¿D\¿¾Í·>¸ÝV?pü?wY?ù>¸}=£È¾ç]9¿â¢m¿Sç¿4qt¿$R¿½ ¿|dÍ¾&6)¾n;=ð->Ù²ê>5?M=?hV?3éh?`^u?8º|?ÙË?V?­|?ñv?¥Ro?zÞf?]?§S?åqI?y??Ô4?³*?Ñ ???´
?:?§ù>·±é>?Ú>êLÌ>>×¾>-²>1G¦>o>>§>ÇÛ><d{>ËBj>ÓAZ>cPK> ^=>X[0>	9$>îè>}]>ê>AÄö=µå=9ÒÕ=RÇ=ü>¹=j¬=ªx =ÆY=ÿ=q\=øÇp=Þ`=·P="B= 4=ü(=7i=s=`t=Nü<¾ê<ñRÚ<`+Ë<ÿ½<&ñ¯<wº£<iL¿¨t¿8P)¿Ä(?OS¡>·5¿)>RD?%~¿;¿eÙ\¿P(\¿~¿T3>¿|½>:n?O´$¿v¾8ÌH?£kv¿þ}?À©z¿Pa?F¿:¶½}åM?7Õn¿?¯J=æ?r?½¨ì¾%_¿8«Ö>×x?Y¾ô<Ôäk¿¤@¿q@>xrj?"b?v¸w>À¾¿Yès¿ïn¿Æ+¿ÕÓ*< å?q_?øÑ?-¨k?0?±k¾>ÜÈà<÷¾4¿ÎbG¿j¿
{|¿]{¿à7v¿ c¿ô
J¿ ,¿¬¿¡~Ô¾M¾Ó5#¾E`#½s/=½I3>ö>Æµ>o:Ý>F ?ç?{ù?bR*?>55?'È>?ù.G?ûN?ªúT?¸Z?_?"Çc?Àg?¼j?Om?¢ÿo?­r?øs?
u?Êõv?b*x?6y?z?>çz?{?Á,|?º¯|?6!}?}?ºØ}?"~?vb~?Ö~?ÎÉ~?[ó~?W?6?Q?âh?"}?«?Û?«?f¶?CÀ?ÎÈ?3Ð?Ö?(Ü?õà?å?¹è?×ë?î?âð?éò?F¿B9q¿3`}¿óÔ½ãÃh?6~¿µb?×À½u¿§Ãæ¾ý"1=þ¨=ôÎ¾Dg¿Oy<¿ÜÄ'?÷
?]ª¿}.O?¿@¿zµ>J6Ó¾êE%?dol¿ìv?¥Cè¾¡kÿ¾ëß~?·OG¾séo¿@ÖÅ>Yr?3¾a~¿æôî¾0?U~?òÑí>ç$Ò¾à3t¿hc¿Í´¾[R >MWP?üÞ?b³^?	×?ù;=/¹¾k4¿³ðj¿<¤¿C-v¿RQU¿, $¿QÖ¾$;¾É@=y>k'ä>>?ÎZ;?ÙéT?zÁg?V t?8V|?Á²?^z?è_|?£w?rèo?g?ØB^?2gT?86J?ã??Ñ5?_r+?ó!?ó?ð¶?ß?­àú>µÚê>*­Û>VÍ>çÐ¿>Õ³>#§>ì>Ñh>ð>·|>mk>i[>sdL>º_>>K1>%>Ë¹>& >L?>9ø=Tðæ=o÷Ö=_È=>=º=UW­=U¡=÷&=¾=?=ûr=øHa=|¨Q=
C=5=.ú(=c@=³V=½.=.vý<Þë<\Û<òBÌ<+¾<Eã°<È¤<¶ð=Rn«¾>>/~?î!Õ¾Cñ=eÐí¾:ß~?§ñ®½ d¿°Â¿µ!¿ß5s¿OnÜ¾>-?$]A?ý{Y¿ðRQ=F_?rù\¿_o?Ñ3i¿\C?oOÄ¾Î"¾«$d?Þ]¿ÔÐ¿½{?u¤²¾æl¿£>r$}?u$û=¥hb¿lM¿Cé=/µb?qi?£>Vì¾Æo¿@ s¿úÂ¿Vÿ½²ü>8ÖZ?¸X?³n?Eñ5?Ó\Ë>÷®X=kx¾o¿D¿A~h¿@°{¿^·¿³Dw¿-e¿L¿=t.¿è#¿ð|Ù¾³ê¾<,¾¨ÏF½C¸=¹+>á>Ù©²>oÚ>þ>Üæ?àý?9t)?q4?K>?LF?N?hT?5Z?_)_?¸zc?<=g?Àj?[m?ñÓo?ºùq?×s?tuu?ýÜv?ßx?\#y?åz?9Ùz?y{?9"|?¦|?N}?°|}?ËÒ}?c~?^~?û~?wÆ~?vð~?Ö?W4? O?Bg?¹{?r?Ì?ª?µ?¿?5È?¯Ï?)Ö?ÅÛ? à?Ôä?yè? ë?[î?¸ð?Åò?×wó¾s×T¿wP¿ÎI%?ádÃ>½b/¿ûRÙ>ú>/rc¿À$V¿ÌÛ¾Ô¹¾Õs,¿'¥¿n[ä¾n ^?ÑÙr>váp¿ p??È4¿çr?¿lªI?7Æz¿wg?ªí¾ù: ¿Öx?ný½qy¿>Éôy?âåJ½/z¿Á
¿ï?ÄÍ?m?\g³¾o¿Á`i¿_Í¾Nv>`cI?M?{c?²½?Ý×=_Ô©¾©.¿kh¿¦6¿áÅw¿cX¿×(¿¬)ß¾óM¾¾ú<êåy>ÿÝ>êÃ?ë,9?BS?çf?	Üs?ì{?Á?ó?°|?@xw?¨{p?d0h?Ðø^?ã%U?QùJ?¢§@?3Z6?1,?TD"?¦?Îb?D?Sü>Oì>ÃÅÜ>×^Î>[ÊÀ>f´>Òÿ§>¡º>N*>F>Â	~>÷»l>3\>txM>Fa?>Ý;2>ù%>¢>Êâ>«ô>*hù=+è= Ø=j(É=~;»=D®=y1¢=&ô=}=À=ý]s=}b=@ÇR=ñ'D=ó6=`á)==ó=é=Ñþ<^!í<Æ«Ü<ZÍ<X¿<cÕ±<}¥<64a?t??°X?}C?òl¿{:?QËg¿Ö__?Áûê>¼G¿æ5g¿¹n¿Þ2=¿dW½ e?Åü>P²x¿cY­>,{±>A5¿£ZS?{nL¿K±?RØM¾³\Ü¾ (t?À§G¿Pp¾ã?©Ák¾ñu¿t1]>¯?~v[>:W¿µ&Y¿YW =cÝY?Åo?HÚ¶>coÒ¾Usj¿c½v¿ü¿;<½×ê>L×U?þ~?¶q?);?ì&Ø>½j =¥Á¾¡Ê¿ @¿2gf¿Ñz¿rá¿Bx¿õÉf¿Ì)N¿ùÃ0¿ä¿ÑtÞ¾;Í¾Òþ5¾:;j½Ï}d=Ù%$>><¯>·¡×>'û>ÑÉ?P?+(?+­3?¯o=?ð F?M?T?ÐY?8Ñ^?ì-c?dúf?Hj?y(m?¨o?Óq?þµs?¸Xu?Äv??ÿw?y?£ýy?!Ëz?A}{?£|?m|?[}?Íu}?ÓÌ}?8~?Y~?~?Ã~?í~?P?(2?¼M?e?Nz?8?¼?,©?Î´?â¾?Ç?+Ï?¶Õ?aÛ?Jà?ä?8è?hë?+î?ð? ò?su÷>Q½æy>²}?³¾5ûh½¯zV¾PÔe?OÇ¿ì¿iQM¿¬Û;¿Éj¿¢]q¿0Æ½¶|?$½ùbM¿-?¢\¿Ñ>?ËC¿e?Ðû¿ñO?Æ¾Wõ<¿=³m?Óç=~¿ø(>r~?N=ßs¿ µ¿ÔPë>îÕ?»J?5è¾Þ(i¿9ìn¿Ûå¾Ù`>B?òF~?üég?Pt?P>y¾å)¿e¿£~¿Õ:y¿»Z[¿M,¿cíç¾­Ò^¾j-e<¶j>íìÖ>® ?¼÷6?üQ?]e?}s?q}{?Ùq?Iµ?wý|?Èëw?Eq?Öh?ù¬_?ãU?.»K?²jA?Á7?üî,?'ý"?X?O?4'?}Qý>+í>ÞÝ>jgÏ>ÃÁ>Ëì´>îÛ¨>>³ë>û>Y\>iøm>Æ¸]>eN>Æb@>,3>Ù&>s[>i¥>ª>ºú=äeé=ÌAÙ=p9Ê=º9¼=È0¯=Ý£=SÁ=<=Ùq=ý¨t=)±c=æS=Ø2E=[7=È*=¸î=3ç=v£	=ö =-dî<1ØÝ<rÎ<À<Ç²<j^¦<(`?ª?X:t?
=C½o¿æ?VRz¿Wá>ª\?QÛg½)è¿pé-¿WÌ¾¡ª>Ë~?¨Ï3>ã>¿Ð?÷£=¾Ô¿O+?ò¾%¿®â>49¼¯Q¿Ý}?=·,¿ú¾¾À[?g4Ý½}|¿,]á=£¬?Kº>nJ¿1\c¿:¾½öO?àu?¡fÓ>bÞ·¾Àd¿ÃÄy¿í'¿õ3í½ôÏØ>aP?}?t?t8@?Çä>TcÔ=¯ði¾Åû¿J=¿:d¿Ýy¿ù¿A1y¿HZh¿0+P¿½3¿ð¿fã¾Ç« ¾^]?¾&Ñ½JC=N>9>ºj¬>RÒÔ>)
ù>Ç«?Ï?;µ'?çç2?TÂ<?èhE?ýL?@S?rjY?¤x^?Ààb?6·f?*j?«õl?á{o??­q?µs?Õ;u?ÿªv?éw?Áýx?Kíy?ö¼z?ùp{?þ|?4|?]	}?án}?ÔÆ}?~?U~?4~?º¿~? ê~?È?ö/?ÕK?ùc?áx?ü?ª?>¨?´?0¾?Ç?¥Î?CÕ?ýÚ?óß??ä?÷ç?0ë?úí?dð?|ò?Æo?9:E?Àk?<?Ùe¿!?ìB¿Ì1}?»wû< d¿å }¿	u¿û¿rd>¿>|?ZÄ¾<¿w6{?ì<v¿xa?1¦b¿ÒÚw?¢ß{¿Ñ?2?Ì­Ñ<Ü6U¿{°]?ÖO>¿÷¿çM=¾ÿ?iy>±k¿Ù -¿ÎòÅ>÷m~?Ñm#?âg¾ºYb¿R¶s¿¨xý¾Ê¨/>ØA:?Í|?
ýk?Bø?	¯0>Aö¾KY#¿ËÍa¿LÜ}¿èz¿6^¿#b0¿kð¾^p¾áÂ'»E3[>>Ð>©6?V»4?ËÜO?X!d?¹@r?¯{?
J?_Ì?ÈF}?8\x?Hq?yi?S_`?ÈV?Ï{L?¾,B?yÜ7?D¬-?lµ#?­
?p¹?×Ê?*þ>VSî>öÞ>»oÐ>¢¼Â>×µ>ç·©>bW>­>ý¯>iW>Ã4o>Fà^>G O>:dA>94>¹'>=,>h>Z_>øü=¤ ê=ôfÚ=sJË=ô7½=~°=@ê£==û=¤#=ûóu=?åd=ÅU=½=F=Ã|8=Â¯+=âÅ=r¯=Ò]
=fÃ =ü¦ï<ß<©Ï<°!Á<¹³<º?§<äË=¡5#?DÇ>$-¿Wzä¾"2M?rò&¿/¾á?rZæ>7¾æ9¾£@<;"+?»Ûv?f|&¾q{l¿ÑM?ãE¾w
¾ô>î¾ > 7>e¿7¿ê?I°¿Ñ ¿Ø®z?Â<$­¿	»7;¿}?\`È>Æ<¿êl¿Tâ½­E?/^y?i-ï>¸¾}m^¿4|¿ /¿Û_!¾ÖrÆ>F K?)|?dv?±E?8<ñ>O>f<R¾ê8þ¾9¿«øa¿íÕx¿Éÿ¿ßz¿cÞi¿#R¿rN5¿øm¿³Pè¾6¥¾­·H¾½"=ø>ãKx>\H©>ï Ò>'ö>À?\?iÔ&?Ó!2?;<?3ÐD?ÎwL?\*S?cY?£^?4b?³sf?pÓi?Âl?Oo?·q?Ass?Ìu?Îv?©Ów?Îêx?ÝÜy?¸®z?¡d{?L|?î|?T}?ëg}?ÌÀ}?Ì~?P~?H~?U¼~?¯ç~?<?Á-?ìI?Qb?rw?¾??P§?3³?~½?gÆ?Î?ÏÔ?Ú?ß?óã?¶ç?÷ê?Éí?:ð?Wò?íK?â´x?°Àd?³µ<Qu¿#y?	¿á:?6?y¿bÞq¿ªÀ|¿6Th¿ÒòÜ¾|?£F`?ð¸&¿Rý«¾Äd?Óï¿Ägw?4w¿Õµ?ún¿Pï?=zB>lh¿K"I?©>}¿¼h½j~?@È}>s@a¿u?=¿äK>å{??Å0?r[&¾·Z¿.»w¿L2
¿<°ü=02?&àz?³o?÷F#?ÔÙR>·v¾{z¿§j^¿Ãï|¿ê¸{¿Çö`¿ï'4¿ö2ù¾K¾õ¼ÅK>>É>üe?Ñw2?N?mÞb?Áiq?]z?T?4ß?t}?Éx?¯%r?øj?Üa?úXW?2;M?ÄíB?Z8?Þh.?"m$?R¼?4d?-n?ZÀÿ>Âzï>¥à>ÊwÑ>vµÃ>Á¶>½ª>%>:n>Üd> >qp>³`>´P>¡eB>X5>ü(> ý>*>¬>Ô]ý=_Ûë=Û=r[Ì=*6¾=1
±= Æ¤=ª[=º=oÕ=ø>w=Tf=#V=¢HG=*u9=ó,= =±w=.=Õp=Ëéð<1à<;¡Ð<Û%Â<½«´<!¨<gÅM¿ª¯r¾Úå¾áï¿¡>}k>Æy½ö.¿TIT?rV?Á§>×">+×>ïf?K^N?ôÞö¾UDB¿Îq?Àæ¾4ÿµ¼É>¼¾ÐB=ó¹>aT¿9W{?ÃÖ¾È^¿lq?¦Û>Ît¿qòÕ½px?Jó>f,¿]Ks¿Ð-=¾´+9?b|?U
?ê¾~W¿?~¿ïá7¿HÜK¾Ç³>à+E?£z?ávx?<ÕI?ý>ýð>Ùi:¾Zô¾êà5¿¡_¿ºw¿	ô¿Náz¿4Vk¿sT¿7¿åÑ¿j4í¾h\ª¾R¾0ª½=^> [q>)$¦>-Ï>%þó>Àl
?ú?·ò%?ïZ1?ce;?Ó6D?~ñK?î´R?ÚX?4Æ]?GEb?Ü/f?ji?Ll?ì"o?ü_q?¡Qs?u?|xv?²½w?Â×x?YÌy?g z?8X{?÷{?|?Aù|?ì`}?¼º}?~?ôK~?W~?ë¸~?ºä~?¬
?+? H?§`?v???a¦?c²?Ê¼?ÌÅ?Í?ZÔ?4Ú?Dß?¨ã?tç?¾ê?í?ð?2ò?{ºµ¾oÔÇ>¬r>U´4¿4¿kºd? ôT¿}Y>+:j?ok½.¿]rQ¿3ø'¿-\½µ1W?ªN*?ÞZ¿ÌÈE½TÏ<?÷y¿ï?£Å¿Ò|? X¿y9Î>@·²>"v¿t0?cFè>h w¿*ÿ7¾'Cz?ùÆ¯>7ùT¿þrK¿8o>ÎWw?f@=?öàÈ½÷GR¿÷z¿iG¿u=ê)?x? 
s?ð]*?KÆt>ÉW¾Ãx¿ÒÜZ¿.Ù{¿¯Á|¿èc¿IÚ7¿Ù ¿sæ¾J½gK<>UÁÂ>Ç?B-0?.[L?Îa?p?}z?¸ë~?Èí?zÎ}?Î3y?z®r? ºj?¾a?©X?VùM?Å­C?e[9?È$/?H$%?m??6	?{ ?È¡ð>÷$á>Ò>®Ä>øª·>oo«>²ó>[/>©>µ©>.­q>/a>ÛÇQ>ûfC>mü5>àx)>¼Í>$í>ùÉ>ª¯þ=í=7±Ü=nlÍ=^4¿=âö±=ÿ¢¥=Ó(=y=8=óx=hMg=FBW=SH=m:="~-=5t!=ï?=Ò=C=,ò<m]á<Ì¸Ñ<*Ã<Ûµ<[©<ïTo¿f²k¿µx¿mU5¿)j\?óå¾?Ç(z¿V Î>?MO;?f3?0A??¡?Å©
?È!?¿LÉ¿³?Êà,¿Gl>±º<ÊÓ,½³ã ¾Èi?ÔPj¿óåo?§Æ¾x9¿ø#d?¢>[Õ{¿mW¾¿pp?â?b¿êx¿¾ûd,?üÃ~? ?J¾}øO¿ÂG¿!Ã?¿êûu¾LÕ >,??!x? Jz?
aN?(Ì?j«7>w|"¾Ô]ê¾$2¿¦5]¿2v¿XÖ¿¢{¿¥Ál¿%öU¿U¼9¿¤0¿ò¾=.¯¾¼^[¾ßÚ»½Ù$Á<Â>Zhj>,þ¢>MXÌ>+uñ>ÆK	?ª?%%?<0?Ïµ:?ÆC?jK?ö>R?Õ6X?Xl]?ùöa?¯ëe?]i?º[l?ön?9q?Õ/s?Iät?	_v?§w?Äx?À»y?z?¿K{?»ì{?>x|?#ñ|?ãY}?¤´}?D~?aG~?a~?}µ~?Âá~??O)?F?û^?t?>?l?p¥?±?¼?/Å?Í?åÓ?ÎÙ?ìÞ?[ã?2ç?ê?fí?äï?ò?!|z¿KU¿n#¿ô¿ÙÉ[>yÃÊ> c§¾iÉÂ¾ß=}?0ñå>ä!¾¹÷¾e\¾ª>Ûz?¸Á>§Oy¿Vy>Ø?M/b¿=#z?ï{¿~No?IÆ:¿y&p>aOÿ>Z~¿Â(?wp?¼ôl¿Dk¾Å	s?5èÞ>/ÚF¿4 X¿õK>Î³q?¸ÏH?d½]I¿Éh}¿åó¿¥v×<¼ ?Ê°u?`v?Á:1?45>ÍP7¾wU¿ù$W¿»z¿¦}¿ª"f¿Éx;¿®¿q¥¾^»U½ÚÅ,>0ó»>-±	?ÁÛ-?áJ?D`?M©o?y?7µ~?ø?Ø~?ðy?¥4s?Wk?vkb?ÕÈX?:¶N?¾lD?:? à/?ÞÚ%?>?¸?ñ³	? ?gÈñ>õ;â>Ó>z¦Å>¯¸>þJ¬>¯Á >dð>cÎ>ÄR>?ér>QVb>ÛR>HhD>wì6>¼X*>r>­¯>B	>¼  >ÃPî=QÖÝ=f}Î=2À=ã²=\¦=úõ=8=9=íÔy={h=aX=i^I=öe;=Qe.=^K"=-=å=²Ë=goó<Öâ<^ÐÒ<2.Ä<ø¶<«ã©<XhS¾lí\¿ßYO¿Ì:<z?Ôk¿¿îq?ü¿l¿-ê¾»d?³1w?{
`?Qu?vq?¤¯R>î÷l¿ªh¾Kx?+Y¿¾Íï>óâY¾È5>cÜµ¾6/?BÜx¿¼æ]?wjþ½ÅP¿w¬R?+ÈÁ>'Ût¿wÚ ¾	wf?Ð9!?á4	¿5ì|¿_¶¨¾oÇ?¨Û?wp?ÓO¾¶àG¿Té¿?G¿ËÕ¾²£>?»8?%[v?NÝ{?¿R?½
?OGQ>±w
¾ÖCà¾T.¿µZ¿XFu¿¸¦¿pT|¿¤ n¿ÑW¿Vè;¿¿¨æö¾û³¾«d¾Í½ÇH~<\Iü=)rc>lÖ>É>;êî>×)?n?µ,$?»Ê/?~:?C?ãJ?tÈQ?VÏW?]?L¨a?.§e?!i?æ'l?Én?ñq?Þs?ÎÆt?tEv?nw?_±x?«y?z?5?{?Ýá{?Ôn|?úè|?ÑR}?®}?öý}?ÈB~?f~~?
²~?ÅÞ~??'?!D?M]?s?û?U?~¤?Á°?`»?Ä?Ì?oÓ?hÙ?Þ?ã?ðæ?Lê?4í?¹ï?èñ?Ï3¿o~¿8¿©Ë-¿R?â¹¾F>©V¿'CB?äU?bc>_T½-l>ë*?=Ð|?#^E=e¿à«?ãX>Î§<¿CWf?àk¿¼W?4¿þq=ãv"?ã¿Ù¦é>(,?!$_¿lÙÌ¾i?£ç?¥7¿Å.c¿°=´j?ÍdS?=×$?¿ð¿F0*¿k·¼?¹pr?)x?Û7?Þ>ä^¾ô¿ÓCS¿ .y¿òe~¿»h¿?¿R5	¿X¾»±½Ñ5>)µ>NÍ?f+?ª½H?í^?Ü¿n?þx?Óy~?-þ?G~?öþy?1¸s?Úñk?c?{~Y?ÜqO?®*E?òÖ:?0?ã&?Î??b?^V
?{±?îò>¡Rã>dÔ>ªÆ>:~¹>j&­>¡>W±>
>Äû>8%t>}c>/ïS>iE>wÜ7>8+>!o>1r>4
> © >lï=gûÞ=ZÏ=¼0Á==Ð³=·[§= Ã=|÷=Èê=å{=µi=ÂY=KiJ=\^<=L/="#=kÐ=@G=!y=4²ô<?¶ã<ïçÓ<]2Å<·<üÄª<B86?öâ½	Ø½9÷;?Ø?JCu¿SLs?|¿Pº&¿>ª?Ûy?Ú§?Cê}?>?§!¾à³¿]7²=º[?:èu¿ê-?mkß¾úÆ>FÑ	¿ï\O?\¿v×E?t,ñ<ìÉc¿Pw=?Róú>ûj¿'Ô¾Ô/Z?3? ì¾I¿[tÌ¾d?7ß?<K*?é8°½Ù<?¿ï¿6SN¿(l¤¾tt>C#2?VÙs?0}?oîV?B?jÀj>ø½ä½äÖ¾"q*¿íX¿£îs¿+e¿÷|¿so¿:¢Y¿î>¿BÞ¿ç´û¾UÄ¸¾Wòm¾9%ß½Bô;ë	í="y\>ò¬>¨Æ>]]ì>ó?G?hH#?n/?rT9?®fB?õZJ?hQQ?\gW?[·\??Ya?Xbe?åh?Òók?Ïn? êp?»ër?-©t?¿+v?!{w?x?Ky?uz?2{?ñÖ{?\e|?Çà|?¶K}?\¨}?¡ø}?*>~?fz~?®~?ÄÛ~?é?Ñ$?.B?[?£q?·?<?£?î¯?©º?ôÃ? Ì?øÒ?Ù?;Þ?Ââ?­æ?ê?í?ï?Âñ?¹¸`>x§.¿Ö4¿Êós=LM}?ÏU¿[R?.æ¿Kè>%?¯
'?!ýÇ>Ð­?åÏf?Ã]?V3¾ tk¿%°=?N<ñ
¿Ó¦E?paP¿í7?õØ¾Sþò½¿Õ@?y´{¿ø1¦>4ÒC?üäM¿Æ ¿@K\?©?°%¿{l¿¸FÃ»bb?uò\?vxÅ=®4¿¬â¿_õ3¿w½?NÂn?óÍz?¤<>?ñX¬>yî½¯¿:O¿w¿2¿ÏÛj¿²xB¿"P¿ÿ£¾ê{®½<>9®>Lã?H$)?äF?]?NÐm?¥mx?9~?ýÿ?~~?ß_z?9t?hl?º¿c?2Z?=,P?çE?r;?]T1?VF'?Q~??|ø
?L?pô>øhä>eÕ>¤Ç>gº>±®>R]¢>2r>7>µ¤>au> ¤d>ÁU>¼jF>mÌ8>\,>É? >¯4>Æé
>R>Æð=x à=KÐ=ç.Â=ç¼´=8¨=D=u¶==Ûj|=éj=Z=,tK=ÀV==®30=¯ù#=¨==&=õõ<¨âä<ÿÔ<6Æ<3t¸<L¦«<iÂy?+;?Êu5?©?ÝF¾.Ê¿ð?§îå<µ t¿5o=þA?Ä©k?GSZ?CwÝ>©Éÿ¾1u¿çöÈ>ÿé+?õý¿½W?²"¿Æ´?Æé2¿h?1~¿i`(?²É:>Q#r¿êä$?9!?Ã@]¿Æh¿ºK?Ð<C?ÑÃ¾Öþ¿&ï¾±L?¥Î~?Û5?+õ¼ò6¿yZ¿qúT¿®·¸¾SAM>yN+?sq?èA~?!îZ?=O?@	> k´½B½Ë¾Ù{&¿¢vU¿/r¿¸¿N}¿¹p¿si[¿	*@¿ù,!¿Û= ¿X½¾U4w¾$Äð½Ï8ºÇÝ=[}U>Ç>ÍÃ>Îé>ã?7ÿ??c"?U7.?ª¢8?¢ÊA?>ÒI?ÓÙP?èþV?9\\?Ñ	a?.e?l©h?|¿k?Tnn?Ãp?mÉr?ft?èv?·dw?x?py?efz?ñ%{?÷Ë{?Ù[|?Ø|?D}?,¢}?Dó}?9~?`v~?«~?¿Ø~?L ?"?8@?êY?+p?p?!?¢?¯?ò¹?UÃ?wË?Ò?Ø?âÝ?tâ?jæ?Øé?Ðì?cï?ñ?p?7>Ì©=¬éB?o ??K~¿¶¿?PN¿Ã/¾sÕd?gn?¯A?¤TS?B?ôw!?F¿A@¿eg?¾Ç ¾Qæ?Ç*¿Å?Ðw¾,¸¾ÙïY?Wq¿Ç©>>¦X?3{9¿4i¿J	M?+.?ê¹¿ct¿)±½AËX?Éle?âª$>Ó7)¿Oé¿W<=¿
¼ô½Ô??6§j?â|?E]D?Ô ¼>®½¸_ü¾¤K¿hÞu¿½w¿m¿YÙE¿¼]¿b¬¾Þ9Ð½ôû=çN§>Ió ?¾&?¼E?,\?ªÚl?«×w?dô}?ý?þ±~?¨½z?b·t?Á m?gd?4åZ?ZåP?r£F?O<?2?7û'?¨-?e´?K?qæ?Ú9õ>û~å>"Ö>fÈ>ËP»>ÕÜ®>ø*£>õ2> ì>M>ßv>ªËe>CV>ãkG>X¼9>ø,>k!>(÷>>\û>® ò=Eá=8°Ñ=-Ã=©µ=f©=g]=mu=TN=Ïµ}=ªl=:½[=L=$O>=Û1=ÖÐ$=å`=õ»=ýÓ=Î7÷<æ<Ö<³:Ç<Pf¹<¬<ÃW¯>(à{?^?÷%?AÄG¿ì=êa5½zO?Õ/w¿âå¾Ùº>`(?­{?±Â`=½§F¿U£N¿¹(?(Û>ùÿv¿ms?-L¿:¶>?}T¿êcx?hÏt¿¬O?©>÷{¿÷d	?Óq0?ñL¿ÑH¿<;?ÇQ?d+¾º
¿ü%¿u(ã>«|?Ó@? ×<oi,¿c*~¿º1[¿ ¯Ì¾4½%>6?$?Nþm?¸?K½^?]í?¬>Pÿ½;SÁ¾t"¿d¹R¿q¿c¬~¿+~¿>òq¿¯&]¿?B¿/v#¿w¿GÂ¾l8¾8/¾Ê¼Î=ê~N>ôT>RðÀ>è=ç>V¾??û?<}!?pl-?(ð7?í-A?ðHI?µaP?úV?« \?º`?¯×d?òlh?åk?@n?ip?ó¦r?ymt?ñ÷u?0Nw?wx?xy?µWz?6{?îÀ{?IR|??Ð|?d=}?ô}?áí}?Ú4~?Ur~?§~?·Õ~?«ý~?F ??>?5X?°n?)?? ¡?F®?9¹?¶Â?ìÊ?	Ò?2Ø?Ý?'â?'æ?é?ì?7ï?wñ?ï¹K?ÇUe?¥P?oµ~?!	½óã/¿ÖD?÷è§¾y!<¿tÛ?iV~?7/w?ô{?kq?FÂ>IZS¿þ¿6}?µ¿¤½HË>¹Èù¾}âÀ>®^Z½é÷é¾m?£´a¿4=4[h?%7"¿À 0¿Çi;?g@?52ý¾òÞy¿ãW*¾üM?9Él?Ôçe>ÓN¿Õ ¿®þE¿öµ+¾4^ô>O!f?=~?Þ:J?ö°Ì>L[½E(ï¾0°F¿Ùøs¿É¿Üo¿¤$I¿À]¿®#µ¾9éñ½v Ü=g[ >Ïúû>#R$?'C?uÁZ?õÞk?2<w?[ª}?Ùö?·á~?Q{?3u?â´m?e?D[?2Q?D^G?á	=?íÅ2?¯(?Ü?æ\?Ì;??Ú^ö>ªæ>¢×>ñÉ>Ð9¼>Ô·¯>ø£>¡ó> >iö>Øw>òf>µ)W>ülH>8¬:>Ú×->á!>¹>8T>5¤>F;ó=jâ=!ÁÒ=3+Ä=3¶=»ð©=*=d4= =Â =¸Qm=õÛ\=ìM=G?=2=ý§%=")=Pv=j=zø<w;ç< .×<Ý>È<mXº<ëh­<r¿Rã>( ?äpÍ½*K¿Â:?Å­#¿×q?ì-¿³ÃU¿Fé½U4>N5>©¾õ7s¿js¿!;\?¼>Ð[¿­y?	k¿Ów_?#m¿Ì¢?µc¿'Á>Oñ>À¿ïèÖ>½F?æ9¿'d.¿ÉÞ(?^?áº^¾Ýn|¿ý¿xÂ>àvy?E÷I?¾Þ¨=G"¿	`|¿Íõ`¿gIà¾ïû=â÷?Ï¦j?=¢?[b?m!?d>þÿ&½Ñ¶¾Í[¿sèO¿qo¿35~¿~¿Âs¿ÙÙ^¿oMD¿Ð¹%¿7ù¿°Ç¾ÖÓ¾ãù	¾ç¼8¿=ç}G>&>À¾>\«ä>¡?aö?` ?Â ,?í<7?@?¿H?éO?,V?±¤[?×i`?Üd?/0h?Vk?±n?sp?Mr?fOt?ØÝu?7w?kcx?ygy?óHz?j{?×µ{?­H|?ëÇ|?-6}?´}?vè}?)0~?Dn~?¤~?ªÒ~?û~?ý?D<?}V?4m?ß?è?© ?p­?¸?Â?aÊ?Ñ?Ê×?.Ý?Ùá?ãå?cé?kì?ï?Qñ?ú¢½;d? hx?\Ü?­ç;¿t½Û0g>Ë¥>5º{¿âr=b$S?e{?~ox?~Æ>?3çC½jw¿$9^¾¨A}?Ä:¿d0B>ð->æ¾ñ=8>>Gk¿"Áy?rL¿Ë½t?mt¿@D¿Á '?ìLP?°¿Ò¾À}¿+{¾rB?þr?V>ÎÒ¿Þ}¿@6N¿¯¥\¾O¼ß>¨2a?t?pÓO?Ü>¬0³¼Î»á¾1B¿»êq¿tö¿Mq¿8ZL¿ÍO¿Q ½¾ÏÃ	¾k?½=x_>ö>Mß!?è0A?kPY?7Ýj?>v?t[}?åë?Ä?Øo{?¬u?ÊFn?I°e?ÉE\?ÄSR?	H?ÏÃ=?§}3?@c)?î??üÜ?e?r÷>ªç>Ï¨Ø>D}Ê>¨"½>¯°>ìÅ¤>5´>éT>,>"y>h>=X>nI>;>·.>±">	|>k	>M>Øuô=ã=ÒÓ=U)Å=Õ·=Íª=¨÷=Yó=Ü±=Ú%=Ãn=®ú]=ËN=é?@=5é2=$&=^ñ=©0=Ø.=g½ù<ßgè<1FØ<CÉ<J»<;J®<)0¿®ëç¾	w¾¸I¿xÜ-¿?Þcy¿Ð¶v?éM:¾®¿(Â¿R[A¾w¾ç´*¿ µ¿;|¾*æy?°¾2Ö/¿ËZ{?G|¿ábu?}Ò{¿%}?JlK¿pÏ`>ßý?½~¿7>êX?b$¿q|A¿Ñ?£Ni?«s¾/x¿D'¿ø¡>q5u?÷S?´¨>ÿ²¿üy¿¢Cf¿}ó¾N¬=úz?ô	g?Rð?³Æe?ÄÎ&?Tz§>¤Ò¼/8¬¾ê1¿M¿Ëm¿1¬}¿ç~¿~>t¿ß`¿SF¿Ê÷'¿	Q¿È¶Ë¾Ol¾èÁ¾«çÊ¼<ì¯=gz@>yö>g1»>öâ> r?ð?«®?JÔ+?ø6?ò??4H?áoO?°ÂU?KH[?K`?´Kd?!óg?ô k?äm?kKp?|ar?-1t?Ãu?Ì w?°Ox?\Vy?:z?ÿz?²ª{??|?¿|?ì.}?k}?ã}?q+~?/j~? ~?Ï~?`ø~?°?F:?ÄT?µk??É?°?¬?Æ·?tÁ?ÖÉ?Ñ?a×?ÓÜ?á?å?(é?8ì?ßî?*ñ?½a¿Õ_->Ùã>ýG¾¥ÿ¿ ?ãÑ¾gD?êËm¿å¾¥·í>¢ÛL??cI?¢ûÝ>YÍ¾Ú-¿µNÇ=Dg?I­b¿¦®Ü>´û½`¿½±8È¼¢Z¥>»ó<¿Å?6w2¿Iv¾v|?ì0Ù¾GâU¿ªè?ì^?Nz¦¾wº¿¥¥¾¬õ4?(x?²>rÏ¿¶%{¿JÝU¿¾¾Ñ¤Ê>Ý[?¾?%U?ì>:²<MÔ¾·=¿f´o¿þ¿±ìr¿½yO¿3¿Æ¾Y¾ãÒ=y[>$ð>f?=??öØW?uÕi?Óôu?°}?°Ü?%6?=Ä{?W"v?wÖn?Rf?Ãó\?	S?ÁÐH?ß|>?¬44?g*?Ü8 ?Á¬?Ý}?þ³?¡§ø>¿è>¾®Ù>_tË>S¾>em±>:¥>±t>1	>ßG>Oz>N@i>hPY>oJ>Ù<>5/>&#>q>>¾>Üõ>e°õ=´ä=çâÔ=t'Æ=uo¸=_©«=ÅÄ=M²=c=RË=Î¹o=f_=©O=J8A=`Ð3=JV'=¹=ë=EÜ=2 û<Fé<À]Ù<2GÊ<¦<¼<+¯<íxñ¾ÖM|¿rOe¿5r}¿ÞX;"M?"i¿|3$?\½>×ïd¿¼b¿Ð¿'¿°f¿Byj¿à£=ìÈ~?ctÚ¾ðí¾éÿf?`o¿@g?lì¿)Hr? ¹,¿±±d=7?¢x¿«)>LÓg?­¿JYR¿#þ>år?Y¢<½ìSr¿95¿c­}>dëo?au[?coF>´¿?w¿ok¿z!¿7=Ë?Í(c?ãü?`ÿh?ô,?lÀ³>R»Ù;Ì¡¾a÷¿J¿Ll¿g}¿=¿cQu¿«!b¿áQH¿0*¿Ú¤	¿¨fÐ¾»¾¾-r½ï =t9>âÄ>NO¸>¼ß>uJ?ûé? Æ?
+?KÔ5?ÜS??s©G?*öN?UXU?yëZ?_È_?9d?Êµg?ëj?*¶m?!#p?>r?Ît?C©u?ï	w?Û;x?*Ey?4+z?¢òz?~{?O5|?#·|?¢'}?}?Ý}?´&~?f~?ù~?Ì~?µõ~?a?F8?S?5j?G~?©?·?À«?·?ÒÀ?IÉ?Ð?øÖ?xÜ?;á?[å?íè?ì?²î?ñ?i_¿!r0¿3ÝÉ¾ÑO¿.Á:¿kúx?°`¿ê~?ô ¿B£U¿l»¢ïé>.î>¼We=ìµ3¿7Âi¿ôÒÍ>!å=?Ygz¿!%?ÿ¾1p>á>i¾aþ>X¿¤v~?³3¿¹¸À¾Ü?á!¾«d¿Iõ>È¶i?Aq¾0Ê¿êË¾â&?¨Ö{?ù\Ñ>Ì¡ì¾Köw¿pî\¿ýj¾Ð"µ>E$V?ùþ?ò-Z?­cû>;g)=ÇOÆ¾uÅ8¿9Vm¿Íá¿Ì¥t¿ÜR¿!¿ÚjÎ¾D+¾¹|=ÉO>Èóé>æ?²B=?[V?·Çh?óHu?¯|?:É?ÙZ?|?v?êco?
òf?1 ]?½S?kI?5??ûê4?ùÈ*?Qæ ?T?m?VM?fËù>¶Óé>g´Ú>BkÌ>Ðó¾>öG²>i`¦>5>e½>ð>{>gj>©cZ>ùoK>{=>Öv0>¬R$>Ó >Ãs>ª>ëêö=Ùå=ÅóÕ=%Ç=\¹=®¬=â =@q=_=Èp=×íp=8`=ªP=«0B=·4=o-(=Õ=\¥=²=ýBü<­Àê<PuÚ<\KË<Ã.½<Ù°<ªpù>éz9¿ZCk¿¥}¿¨/?70n>«\õ¾ç:¼=ÂþN?¢¿qÿ¿>·c¿³¡b¿0¿Q6¿ò¼Ð>\gj?v¯+¿ÝðT¾¹C?Kt¿}??Ey¿+^? ¿÷à½P½P?,m¿Þ=ó{s?0æ¾äÈ`¿"ðÐ>ÛÆx?àó%=£åj¿  B¿¹â7>mi?±c?x~>S¿ps¿«qo¿xH¿¸·;¶ê?_?íÇ?hl?01?£è¿>w¨ø<FÇ¾©¬¿G¿õEj¿àd|¿#¿cWv¿-¶c¿KHJ¿ub,¿ô¿3Õ¾¾SI$¾Hn'½QM=Kl2>Æ>|kµ>²èÜ>" ?uâ?¾Ü?9*?ç5?´>?ÅG?ì{N?íT?<Z?w_?i¾c?(xg? ¶j?m?¥úo?Xr?Iôs?Çu?õòv?í'x?â3y?9z?¥åz?<{?+|?¯®|?O }?Â}?Ø}?ñ!~?óa~?f~?mÉ~?ó~??C6?JQ?³h?ù|??¼?çª?N¶?/À?¼È?$Ð?Ö?Ü?ìà?å?±è?Ñë?î?Ýð?¢}½ '~¿bt¿Â{¿½ø	e?ã#}¿Ò#_?X½(¿uñ¾	ï®<¼Î}=k©¾øh¿Y9¿r´*?ï?Mx¿Q?ùÆ¿º>ÂÔ×¾d'?#Em¿´_v?Üä¾s>¿ ·~?|HA¾ hp¿¡aÃ>Ôùr?Ø¾,ï}¿ö·ð¾6ß?Un~?ü=ï>ÇÐ¾0þs¿½dc¿îµ¾A>	P?ìÚ?Xì^?3?E[=LV¸¾ºÙ3¿Ðj¿7 ¿h@v¿BuU¿Î$¿e·Ö¾Âò;¾ú½==Æ<>ÆÛã>ä`?ÞA;?ñÖT?´g?¤t?Q|?±?á{?c|?	w?ïo?g?K^?ÎoT??J?fì?? 5?öz+?L!?û?­¾?mæ?Àîú>èê>Ê¹Û>ìaÍ> Ü¿>c"³>{-§>aõ>q>>KÆ|>ªk>Ùv[>ÞpL>Pk>>nV1>*#%>0Ã>ç(>uG>l%ø=}þæ=×=¨#È=®Hº=ûa­=ü^¡=20=Ç=?=ß!r=ÔVa=bµQ=)C=¶5=)=J=µ_=7=Èý<íë<ßÛ<OÌ<ß ¾<)î°<]?Íõ=¼¾É9>Òi?ûµä¾ó¡>kôú¾%n?|v½¾a¿ñ¿¿²¦q¿u=Ô¾Q0?çÆ>?b[¿¹=Ï?wz[¿Bdn?$h¿l÷A?=À¾ì¾V e?Êú\¿YôÌ½÷e{?ö¯¾ l¿±:¡>¶O}?îÒ>ña¿±N¿Ôeâ=YUb?Ø¿i?Áö>È/ë¾!Ko¿Ms¿/¿7Ã	½U¹û>IZ?~Q?Õn?A.6?ôðË>'_]=ôñ¾6R¿åC¿§fh¿§¦{¿¥¹¿pPw¿Q@e¿½6L¿.¿7@¿I¶Ù¾#¾c-¾gH½ú=Ûa+>.]>ø²>ßNÚ>Sñý>Ú?ò?5j)?Ìh4?>?F?'N?5T?0Z?j%_?Ewc?<:g?$j?ÀXm?øÑo?øq?Õs?*tu?ÞÛv?çx?"y?*z?Øz?ì{?À!|?0¦|?ò}?a|}?Ò}?(~?Î]~?Î~?PÆ~?Tð~?¹?>4?O?/g?©{?d?À?ª?µ?¿?.È?©Ï?$Ö?ÀÛ?à?Ñä?vè?ë?Yî?·ð?pqN?Ýgþ¾¾X¿öÞ¿¸g!?t	Ì>2¿¨à>ñ}ó>.
e¿×TT¿ÁFÖ¾7´¾(~*¿¿kpè¾«]?Åz>q¿o?D¦3¿8?"ü¿ßH?øz¿g?ç¾Bz¿y?u¨½`ñx¿î~>`Ïy?æT½ù,z¿(
¿b?ôÈ?i?2*´¾@o¿«;i¿ÂÍ¾®>+I?R?§^c?Ï?±ÜÕ=ô3ª¾yË.¿í#h¿Õ9¿Q¼w¿PX¿(¿òÞ¾IL¾Çmý<Ez>d¹Ý>Õ?¥:9?xLS?bf?èàs?Eï{??;?®|?cuw?xp?P,h?bô^?>!U?ôJ?Ú¢@?uU6?],,?Í?"?¥¡?^?A?°ü>üë>è¾Ü>^XÎ>BÄÀ>ªü³>pú§>µ>%>A>{~>:´l>ù\>µqM>ûZ?>ý52>¢ó%>>Þ><ð>ç_ù=m#è=uØ=½!É=F5»=F>®=,¢="ï=Þx=´»=åUs=ub==ÀR=k!D=à6=¹Û)=K==ä=Èþ<zí<n¤Ü<¯SÍ<û¿<xÏ±<ib?,)^?^	?cÁU?î´F?EËj¿¶v7?;
f¿2a?#µä> ¿£h¿
o¿m÷>¿Ó~½Dd?; ?A8x¿÷à©>>±´> a6¿0T?BM¿ø³?pR¾ÐYÚ¾ç×s?ÑAH¿=m¾/r?PÇn¾}½u¿²Ñ_>j§?Ñ/Y>W¿9âX¿.Ø'=Z? o?\%¶>Ó¾ôj¿¨v¿ØÒ¿,!½RGë>q÷U?³~?ãpq?m	;?]××>%=0¾è¿¶µ@¿tf¿ÉÖz¿¢à¿|<x¿Àf¿!N¿µ0¿¿ÍUÞ¾µ®¾!Ä5¾]i½Le=IU$>"'>È¯>G³×>Ùû>ÌÐ??¢(?û±3?és=?¦F?ÚM?pT?ÒY?aÓ^?Î/c?üf?Jj?¶)m?©o?Ôq?Í¶s?lYu?«Äv?Çÿw?y?	þy?zËz?}{?å|?¦|?}?øu}?ùÌ}?X~?£Y~?2~?0Ã~?í~?`?62?ÈM?©e?Wz?@?Ã?1©?Ó´?æ¾? Ç?.Ï?¹Õ?dÛ?Là?ä?:è?ië?,î?ð?bín?qZÌ>M¾*b>u}?ne¾Dú½<¾ _?É=¿Ò~¿«F¿ÅJ4¿9f¿Xt¿-¾Öuz?ýÚ@½|Q¿+~?owY¿r1:?% ?¿¯[c?ûã¿ÌgR?¾ q:¿õn?t¤n=¹)~¿Ýè3>ð%~?Ç(=Ît¿¿¶¿î>Öä?(í?wä¾iÁi¿!on¿ÿã¾re>»B?{d~?Yg?½?°>áë¾ª)¿¬Pe¿µ®~¿Xy¿[¿ã+,¿ªç¾x']¾o ~<l>ê×>\C?-7?Á»Q?Ùze?Å$s?{?Vu?ç²?eö|?áw?Îþp?¢Æh?#_?cÑU?©K?mXA?	7?.Ý,?Óë"?ÔG?9þ?Ó?44ý>»í>¿ÃÝ>NÏ>6¬Á>ÌÖ´>EÇ¨>°u>Ù>ê><>µÚm>]>~rN>J@>3>Ä&>ÙG>$>ÿ>\ú=XHé=G&Ù=ÏÊ=Ü!¼=¯=-ù¢=®=*=(a=êt==c=ËS=ÊE=
m7=Þ²*=Ú=fÔ=÷	=® =àEî<ý»Ý<ØWÎ<À<Ç°²<WØ·¾©¹j?}?§y?ä>ôuu¿~?}¿O7û>ÇU?æ1Ø½?!"¿c¾5¿ïÞ¾ô»>öÊ}?ßS>Q·¿T?ÐbØ=M¿w±/?pÀ)¿ÊRë>Oï¼½Ó¿ê|?ãm/¿k°·¾­?Sèô½|¿ãõ=ÆÈ?v>³K¿¬vb¿èØë¼TïP?L¡t?ÀÐ>cº¾fPe¿ey¿>.&¿ô%å½ÿÚ>YQ?½ }?#×s?²À??åã>¸Ï=¬(l¾p¿Yt=¿Ãod¿Võy¿ø¿|y¿85h¿eûO¿$Ö2¿ÁÊ¿ïâ¾î6 ¾a|>¾Ú'½F=ªF>Yß>õµ¬>ðÕ>Fù>¬Æ?¥?JÊ'?uú2?¢Ò<?5wE?
M?2ªS? tY?ù^?èb?½f?ªj?súl?o?Û°q?×s?>u?Z­v?ëw?ÿx?Ôîy?K¾z?!r{?ÿ|?|?
}?o}?dÇ}?~?sU~?~?À~?çê~??+0?L?!d?y??Ä?U¨?´?A¾?Ç?²Î?NÕ?Û?ûß?Fä?ýç?5ë?þí?hð?¤øN>JGy?`?*?ÉU[?M¨Q?3mU¿ª¾ ?P/¿»¯?g<z½m¿úy¿ãn¿	¿ýõH¿/ßS>y~?h«¾&ò!¿*}?s¿"\?¼6^¿Gu?æ-}¿h7?ÔººQ¿Õ`?»9>^ÿ¿=mò?Ïë>zm¿có*¿%Ì>ÖÁ~?D,!?l#r¾c¿zûr¿©ù¾­7>@;?«}?Yk?¤Á?m+>>¾MK$¿MWb¿ìþ}¿RWz¿ùÀ]¿Â/¿2ï¾&«m¾èv9
·]> VÑ>¸«?M5?Ö$P?pUd?=cr?"{?àP?äÈ?;}?Jx?Eq?_i?SB`?<V?p\L?B?½7?g-?^#?í??#°?MVþ>#î>OÈÞ>DÐ>ûÂ>É°µ>ý©>³5>r>r>Ç;>o>°^>:sO>0:A>õ3>|'>$
>;H>¾A>ÊÔû=?mê=7Ú=ÞË=p½=Öö¯=BÆ£=þl=YÜ==î½u=ð²d=ñÕT=(F=2T8=+=¿¢=¾=c?
=§ =Erï<ÓÞ<\Ï<3÷À<³<b·z¿ói>g/??þ?æ¿b^¿VE]?.:¿»©J½?|¾>l¾Ù¸¾´ìi½ü?Ûuz?`Ü½H:q¿,DF?ü¾mh ¾q­?w8þ¾k¬>\>~2¿ïÿ?L¿»õ¾÷À{?î»ac¿Ð¶¦<³}?1Á>>¿BÅj¿É½+åF?m½x?É²ê>À0¡¾ª_¿Ù{¿ >.¿4h¾xÉ>uíK?Úf|?Uv?'SD?6ï>wÄÿ=V¾~Òÿ¾F!:¿Xb¿^y¿  ¿cíy¿Ùi¿wÑQ¿ð4¿	¿¢ç¾»¤¾ù0G¾²½Jð'=6>ªmy>Ë©>ßvÒ>îö>±»?ú.?/ù&?:B2?¶0<?/éD?®L?|=S?Y?3.^?äb?¿~f?Ýi?÷Êl?ÆVo?q?ºxs?#u?ív?<×w?èíx?ßy?±z?¦f{?|?s|?¥}?i}?ÉÁ}?§~?>Q~?ì~?ã¼~?*è~?¦?.?<J?b?®w?ò?Ä?w§?T³?½?Æ?5Î?âÔ?©Ú?ªß? ä?Àç? ë?Ñí?Að?7¿#¦C?ï?Ú w?0U>z_~¿HWl?·Bz¿JQ?°Kä>¿n%¿Íy¿ ¿å¾q¿V¿l?­Si?ü¿Ì¾ik?U&¿zfs?Ùvs¿,Ö~?{r¿|?>nd¿MN?`y>l~¿ã½=0? f>OÕc¿7¼9¿Åd¨>aa|?q½-? 5¾}\¿Ýv¿Ô¿:Ú	>
4?]{?[Þn?z!?_ëJ>zî}¾fÛ¿M8_¿*}¿v{¿cU`¿ãH3¿5÷¾+~¾Üôy¼ä]O>ÎË>K?Rÿ2?ÄN?.*c?Vq?T«z?,(?3Û?|}?d°x?zr?õi?ðæ`?Ç-W?ÄM?ïÀB?Åo8?	=.?mB$??~<?1H?ùwÿ>
6ï>Ìß>\:Ñ>{Ã> ¶>`ª>õ>DA>Æ:>8Ù>l'p>ôÂ_>ésP>º)B>wÔ4>ßd(>jÌ>Ný>yê>3ý= ë=ßGÛ=êÌ=û½=Ó°=W¤=ê+==¬=ðñv=¡Ñe=ÉàU=
G=[;9=%a,=ùj =I=Ïì
=xH=ªð<ëß<*`Ð<OéÁ<ds´<° 3¿0%¿ô¹¼È¾öcz¿æ=oÃÄ>W¾5
¿y$e?Ñ]C?µò_> ]=Äv¨>îå[?{Z?]©Ò¾ N¿¬j?Ç,É¾VÇ¦½á¥>CA¾¹Ã=Õ+¤>IN¿¿}?r"ç¾P¿t?<	ã=éÏ¿9ò¢½7ly?izé>a-0¿Âq¿_+¾]<?ñ{?ó?¾E)Y¿o«}¿rþ5¿ B¾f&¸>IF?[ìz?þ x?í¿H?«ú>2ð>²÷?¾R©ö¾Í¼6¿è.`¿ðýw¿_ø¿'²z¿×ÿj¿BS¿Ý7¿ñC¿¸ì¾<©¾¿áO¾¦½>	=¡#>Jùr>|ß¦>ÖÏ>çô>Ý¯
?A?Q'&?K1?%;?ZD?ÎL?OÐR?ÂµX?Û]?rWb?­?f?-¦i?Cl?R-o?iq?xYs?lu?c~v?ÑÂw?3Üx?3Ðy?¾£z?[{?ú{?É|?#û|?b}?&¼}?Å	~?M~?C~?·¹~?kå~?E?,?sH?a?Wv?É?Ã?¦?²?ô¼?ðÅ?¸Í?uÔ?KÚ?Yß?¹ã?ç?Ìê?£í?ð?"y¿mx½e!?áõ>Ï§¿Õd0¿÷¹w?7l¿ëÈÅ>@bU?k[¾H¿>c¿L(?¿/¾ÆF?S =?Ï3M¿z¾ÃsJ?è|¿bÜ~?ã{~¿ûÓ~?#`¿oç>>¿r¿xQ8?È´Õ>)uy¿>¾¾Oá{?@+¡>
âX¿qPG¿²>lÆx? 9?7Ôð½réT¿z¿Ôø¿Æ}·=5,?tDy?'r??(?_j>.¢`¾ M¿2ô[¿Ç1|¿u|¿Ñb¿i¾6¿#$ÿ¾°?¾4û¼Øù@>¾ÍÄ>1k?:ß0?äL?ùa?Ðp?¶5z?9û~?Óé?êº}?y?kr?Dj?úa?ÚW?ÀM?ÝsC?Â!9?ì.? í$?8?%Û?ûß?L ?®Hð>Ðà>è/Ò>úbÄ>Qd·>-«>oµ>õ>
ã>v>¨Mq>ÐÕ`>tQ>9C>ã³5>:5)>«>[²>1>Iþ=ý¶ì=¥XÜ=òÍ=ç¾=]¯±=i`¥=Õê=Ð?=Q=ñ%x=Rðf=¡ëV=áH=":=G8-=23!=n=:=Üé=Ëñ<¨á<SdÑ<jÛÂ<²Tµ<$e>Y¿·F¿R`¿*[¿9?8¾+¬Ã>± l¿én?õz?¢³?·wê>ZG*?5|?6!?8«,¿â¿ð}?z®¿_´>üÀ=Þ½ß½Ã½s^÷>pd¿t?K£¾Á¨1¿7¢h?f>ìH}¿­ø6¾cür?­?§ ¿îbw¿dq¾©X0?O9~?$?[¾NR¿nø~¿l=¿wLi¾Ü¦>mò@?¡1y?­Ãy?*M?eû?Iè/>ï¹)¾eí¾>G3¿+ó]¿!èv¿5á¿½i{¿#Ul¿·dU¿ã9¿Ûy¿Ãð¾ä¹­¾X¾¥¶½²Õ<d>Jl>åñ£>§3Í>x9ò>1£	?5S?²T%?©Ï0?ðê:?dËC?iK?ªbR?VX?]?¬b?Q f?oi?Vkl?¬o?ØDq?:s?)ít?½fv?M®w?iÊx?ÆÀy?^z?O{?ð{?{|?ó|?\}?}¶}?Ý~?ÃH~?~?¶~?¨â~?á?ü)?¨F?}_?ÿt??À?¹¥?Ò±?L¼?_Å?:Í?Ô?íÙ?ß?rã?Fç?ê?uí?ñï?Q6­¾3V¿ì/;¾N¶¾fúu¿z$½2D?0¿í(¾ö?Ô6>qîÛ¾H¨¿*ß¾$:D>=q?tÝû>°Æp¿uv>új?l¿úõ}?MÏ~¿Çu?«¹F¿­>ã>PÈ{¿?fÓ?+q¿4F¾v?*¡Í>¬NL¿S¿ð<>võs? D?5 k½L¿N|¿<	¿MÍ5=$?Év?Sót?â´.?«ù>$C¾,¡¿X¿®{¿vU}¿L5e¿Ú":¿A¿Ïf¾]<½¦2>¸{¾>Â
?¹.?\;K?AÂ`?{þo?K»y?
Ê~?Äô?ö}?ûty?s?k?n+b?òX?/pN?æ%D?Ó9?/?%? Ü?zy?w	?Ý ?ùZñ>SÔá>9%Ó>3JÅ>Ü=¸>mù«>(u >¬¨>=>ô>Ïsr>èa>uR>­D>F6>*>åP>eg>å;	>óÿ=ÕÛí=hiÝ=øÎ=Ô¿=²=z-¦=¿©=ñ=ñö=ðYy=h=wöW==ûH=©	;=j.=kû!=Å½=¦G=@=t÷ò<6â<|hÒ<ÍÃ<6¶<Åçp?Ø1¿º¯{¿Ø0t¿Û¾¼?77I¿XìV?~|¿Dé}=°¢v?¹-g?"H?Fpf?vA{?½'¬>%â^¿ïí­¾Y}?ï§J¿3õÃ>Jý ¾3BÅ=Ø¾ãa!?[Zt¿Se?ô9¾H¿tY?»«>ãÕw¿R¾Hrj?6A?À¿Ý{¿²-¾¦î#?ï?qâ?u&¾:óJ¿j¿¿qD¿4¾tÌ>;?7w?ÿN{?%Q?A?ÇG>g¾ä¾îÀ/¿w¥[¿Áu¿º¿|¿¬m¿Á!W¿;¿=«¿¥õ¾O3²¾&7a¾4óÆ½5«<uù >¼f>Æ¡>Ê>WÜï>°?d?R$?T0?G:?¡;C?~K?ôQ?ÙõW?©3]?Åa?¬Àe?®7i?0;l?ÕÙn? q?s?ÆÑt?úNv?°w?¸x?F±y?îz?àC{?èå{?Ur|?ì|?sU}?Ì°}?ðÿ}?~D~?â~?S³~?áß~?y?ç'?ÚD?í]?¤s?s?½?Ø¤?±?£»?ÍÄ?¼Ì?Ó?Ù?µÞ?+ã?ç?aê?Gí?Éï?àë?icp¿Ü^¿Åbe¿ÁVd¿Öô?|<²Aõ<Y*¿¸>e?f0?<¤¾n½h?¯ÿ? I>H¦¿ãMÌ>ÔÈ>AßN¿l¾p?mt¿Qc?L'¿->5?Ï¿ú?¹ ?©«e¿q²´¾Åm?Oø>V3>¿ý^¿Â(á=~ôm?ÅÙN?ü;D;(C¿m~¿ïº%¿ pº_£?_ís?âw?ö4?>½z%¾Ù¿ÝþT¿pÓy¿Õ~¿Eg¿åu=¿Ìa¿á¾/{½$>!¸>]?,?I?¥_?'o?<y?~?ü?.~?<Óy?~s?Þ­k?MËb?.Y?DO?×D?:?[H0?¯@&?Ô?{?Æ
?8m?ëlò>Å×â>QÔ><1Æ>A¹>ªÅ¬>Ç4¡>B\>a3>>±>ás>Vûb>¢uS>øD> r7>ÛÕ*>>i>ä	>%_ >¨ ï=&zÞ=úÏ=¤ÀÀ=Ýg³=ú¦=§h=B£=a=îz=¯-i=MY=óI=Ðð;=æ.=£Ã"=x=õ=¤,=Ø#ô<Ã1ã<¤lÓ< ¿Ä<O·<É	K?à°>Øþ¾êPã¾o~ç>ÀN?jö¿â??¿2âã¾mÜ9?@ý?£îw?Í?h+Y?ØAK;¦þz¿~@V½d¼j?k¿4Ì?©®¾92>£{ë¾ýA?
}¿A)Q?u½¬ô[¿äAG?Ã)â>äo¿i¾½¾Ñà_?ÏX+?-ý¾ºr~¿SÒ¼¾ºÓ?µÿ?Ô#%?äÃâ½4C¿ùÿ¿bAK¿`¾BÐ>R5?Qýt?¢|?ÇU?i?_>nú½Ú¾/*,¿úEY¿©t¿I¿:±|¿dßn¿QÖX¿é=¿Ø¿Aù¾Ç¨¶¾sÛi¾u^×½}4<ÕÃó=³_>&>ÊéÇ>}í>Z?=t?2­#?NZ/?¢9?I«B?J?ùQ?FW?iß\?'|a?¾e? i?Ò
l?Ì¯n?ýûp?Ïúr?B¶t?7v?ûw?¦x?³¡y?n{z?,8{?ÄÛ{?i|?fä|?ÚN}?«}?üú}?4@~?*|~?°~?Ý~??Ï%?
C?\\?Ir?F?¸?ö£?K°?úº?:Ä?=Ì?-Ó?/Ù?bÞ?äâ?Ëæ?,ê?í?¡ï?p?T¾ëp¿WØp¿Õf°¾àÐx?ªJ¿W²?µ v¿ ?Àr?M]ä>¨7->ì§>*L?½q?slñ½?x¿tÓ!?«>8¨%¿ÔÙW?mÆ_¿ØÐH?1¿9¤æ¼ñ1?7~¿$É>ó&8?W¿	ç¾Ác?ïk?«.¿Äúg¿P= Ëf?Å3X?Î=è:¿l¿
/¿UL=½_î?±p?õ¹y?Ç;?õ£>µ¬¾õ¿ÊNQ¿<nx¿¶~¿?²i¿9·@¿a9¿S¾Ý=½Þ>ó½±>Ú`?[*?ìÖG?PC^?XKn?¸x?÷Z~?ÿ?÷b~?Ë.z?¡÷s?Ð<l?ic?ÛÖY?CÍO?JE?U3;?õ0?Éé&?¢$?)µ?Å¥
?3ý?~ó>ïÚã>.Õ>Ç>ð¹>È­>Nô¡>Ä>tÛ>{N>Þ¿t>ÿd>vT>tçE>òQ8>!¦+>HÕ>iÑ>B
>Mü >v%ð=àß=ùÐ=+­Á=D´=Ç§='=úT=ÐA=ëÁ{=\Lj="Z=òëJ=ö×<=­½/=Û#=r2={¢=Î=<Põ<PIä<ÌpÔ<»±Å<ø·<¬¬½«^u?ì#­>­>xSp?íâp>ØN¿»6N?N¯¾­AU¿À£> e?O|?×q?q?4)©¾7~¿Vu>;]F?·6}¿-C?}£	¿£®ø>Æ¿àz\?æ¿¦8?ÏMÖ=ãjk¿ÓÔ1?Ê
?{td¿]Eì¾f_S?«';?Ø¾v×¿§Ý¾
	?;|?ß/?@ûo½ÀÐ:¿ú¹¿Ï¢Q¿e®¾ÑNa>Ò.?Îr?7¾}?éX?÷e?æ*w>«Í½hÑ¾X(¿çÔV¿+?s¿>¿A}¿<p¿TZ¿Æ?¿!  ¿yþ¾2»¾B{r¾!Æç½f;µå=@Y> >kBÅ>ë>2x??TØ"?.?ý8?^B?J?îQ?H4W?Ë\?h2a?@e?.Èh?;Úk?n?O×p?õÚr?t?v?,pw?x?y?Ýmz?j,{?Ñ{?¶`|?¿Ü|?8H}?V¥}?ö}?ä;~?nx~?Þ¬~?JÚ~?¢?µ#?8A?ÈZ?ëp??²?£?¯?Pº?¦Ã?½Ë?¾Ò?ÏØ?Þ?â?æ?öé?êì?yï?Íxï>*?ÚL´¾_Ð¾Æ>Ye?_u¿«ÿq?9äu¿
Ë=¬|?nøJ?¥È?*?Ûu?FG?z×¾ö\¿µpQ?v	Û½Wæ¾C}4?þ·A¿ö '?a²¾èE¾ÚL?Ë(x¿Á>K¿L?Ý¯E¿Z¿/V?)p#?¾Ó¿ño¿(
#½é^?t¤`?ý=Ø/¿´þ¿éê7¿U:»½~ö	?Øm?Ê{?³×@?z6³>3Ó½ø¿ì{M¿Dåv¿z7¿ Ëk¿æC¿±¿§¾ª¼½Ì>ÈR«>¨?8#(?ÑF?Jû\?Øim?N/x?~?{ÿ?¤~?§z?znt?×Él?Ed?Ô}Z?*zP?£6F?bâ;?:¢1?f'?
È?R?<?÷?Âô>ÏÝä>ÐÖ>ÁþÇ>Éº>Ç]®>¼³¢>1Ã>w>«ë>Ååu> e>vU>ÇÖF>:19>_v,>q >d>ê5>r>?Jñ=à=õÑ=¯Â=T µ=¢¨=sæ=°=>ç=æõ|=kk=ö[=LäK=¿==Î0=T$=Èì=æO=lo= |ö<Ý`å<ôtÕ<Ö£Æ<ëÙ¸<ËEb¿íM?o?×i?»ùk?÷wã¾ü¾*§>t>H~¿$¾w?{T?Y??j³>H ¿9h¿\É?Á5?~¿`¡d?>5¿ì['?nB¿	p?'7{¿ö?V}{>ú²v¿«?§"?Y¿V¿ü7¿¼	E?wI?J{²¾Ì¿é0ý¾Óõ>þ	~?õ:?&4Î»ê2¿í~¿Ð¤W¿ê%Á¾6°<>-^(?7Îo?¡~?É\?¯?T>W ½4`Ç¾ÁÌ$¿pRT¿ äq¿Ré~¿Ã}¿'>q¿º%\¿A¿#"¿=¿w¿¾i{¾ò)ø½Cw»¹\×=uR>},>tÂ>ìºè>9h??·"?0â-?ÂW8?àA?¡I?m§P?áÒV?Ï5\?Vè`? e?h?l©k?'[n?v²p?öºr?Ø~t?v?D[w?ex?Uy?<`z? {?VÇ{?ÖW|?Õ|?A}?}?ñ}?7~?­t~?©~?z×~?2ÿ~?!?d??2Y?o?ç?ª?/¢?Á®?¤¹?Ã?=Ë?OÒ?oØ?¼Ý?Tâ?Næ?Àé?»ì?Pï? jû¾õÿ?3ø>Á>vvi?OÍ>No¿øv?Ö*¿¨xã¾L?[Éz??Y?APf?è?LÆ?B.¿­,¿Wrq?ÚPµ¾òh¾Ã_?¿29þ>¹4¾ñ³¾>\a?£l¿¶á>1^?¡1¿*!¿G?S
5?Í¿ÈXv¿ê½#U?¶"h?Æ£;>ü%¿º¿(^@¿¶¾Ñ¿ ?1i?¼+}?sF?IÂ>z½\Ã÷¾æI¿Â8u¿¿LÊm¿G¿mÆ¿¯¾ÝÜ½Cùð=Ðß¤>QÔÿ>®å%?ÚZD?­[?l?Å¡w?ûÚ}?®û?&Ã~?ÎÝz?ãt?ôTm?[¡d?y#[?ø%Q?åF?²<?BN2?:(?	k?ï?øÒ??§ õ>gàå>7øÖ><åÈ>¢»>¦)¯>s£>v>i+>Í>w>3f>ßvV>ÆG>y:>F->Y!>Z;>Þ>6>oò=I¬á=îÒ=0Ã=üµ=¬a©=X¥=f¸=¬=à)~=³l=É!\=¥ÜL=@¦>=îk1=J%=§=Pý=Ï=©÷<jxæ<yÖ<ñÇ<9»¹<_¿ûºè_?Æm?ÿÒ>Fj¿3{´>!¾\\??ëXe¿E¿M>êl?Rß>8©½¦Y¿4;¿þ<?¸+ª>õho¿´Iy?í¦X¿&XK?6^¿Hû{?0«o¿ãò>ùUÃ>¨}¿ºJþ>ÝX8?ûF¿ æ ¿ÿ4?tV?ðE¾P~¿TÍ¿Ü×>Vª{?Ç£C?G<=ÿè(¿)}¿ªD]¿Ó¾3Ñ> ¸!?>Úl?L?¢`?¯Þ?8 >þe½z¦½¾Á!¿É¾Q¿ yp¿~¿Ã8~¿]r¿rÀ]¿ÝøB¿B$¿¥q¿}ðÃ¾`Ö¾ÒD¾v<¼%É=e
L>7>éî¿>*Wæ>pW?Ý?^,!?%-?b±7?Ðö@?¤I?u7P?qV?và[?ñ`?;¿d?¬Wh?exk?0n?qp?Ñr?ñbt?Íîu?CFw?.px?ry?Rz?»{?½{?ìN|?UÍ|?Ý:}?Ä}?ûë}?53~?èp~?Z¦~?¦Ô~?¿ü~?y?=?W?+n?µ?¢?I¡?û­?ù¸?}Â?¼Ê?ßÑ?Ø?hÝ?â?æ?é?ì?'ï?­¿°?öz?Fçm?>lr?µ¾Ï<¿QÕ#?X'¾!U¿ª÷×>rúw?=¥}?|?¿f?Z>,`¿ßÜ¾¾u?²¿4Yú;J«>oÝ¾/¥>±Ò×9ÿ8 ¿=Rq?`C\¿ø#q;ì8l?4¿~ò5¿©6?pE?yvñ¾%'{¿0/A¾»J?t¦n?¨x>=Ú¿1Ã~¿¤\H¿}9¾î>§Èd?Cc~?ßÔK?ä*Ñ>Û¢6½êfë¾`pE¿óhs¿ßÙ¿í¯o¿ðJ¿G{¿L·¾È_û½BÎÓ=We>cNú>y¢#?B?OZZ?k?zw?¨}?2ô?zî~?@1{?QUu?%Þm?Ø:e?ÊÇ[?­ÐQ?G?C>=?­ù2?"â(?¡?9?*i?Ø«?1±ö>µâæ>bì×>ËÉ>S{¼>fõ¯>K2¤>Î)>JÓ>â%>S1x>Eg>,wW>JµH>¯ï:>Æ.>±">Lð>0>³Ó>Âó=÷¼â=ãÓ=¯rÄ=ÂØ¶=´.ª=:d=j=2=Ø]=]¨m=,]=ýÔM=d?=C2=ä%=ta=ºª=2²=fÕø<öç<C}×<È<º<6hk½½zM¿*J>é½>¤¾)¨u¿ê¼X?§µD¿ï|?Ñ¿jh¿¶T~¾'
>js=<»Ý¾ z¿Ç=÷¾5ôf?èõ=PÓP¿þ?°?q¿åÈf?Or¿ÿÿ?å]¿{áª>v?ÿ¿UÅ>	K?/4¿µù3¿ud#?:¾a?\RF¾.g{¿Q¿mG¹>{_x?CL?z!É=X¿uÃ{¿Ìb¿åºå¾º|å=â?¤©i?Ò¾?zVc?ô"?´>á²½GØ³¾´1¿'O¿Äün¿v~¿ ~¿üps¿lR_¿ôßD¿Õ[&¿T¢¿+UÈ¾¾xr¾Jx¼àêº= E>$A>ÐB½>Ëñã>ÚE?Ø¬?IU ?Sg,?a
7?.d@?#H?ÇO?ÖV?¿[?9S`?)~d?
h?%Gk?¼n?Ahp?zr?éFt?{Öu?)1w?á]x?«by?ÈDz?Î{?¶²{?÷E|?Å|?#4}?ð}?ïæ}?Ö.~?m~?£~?ÎÑ~?Hú~?X?µ;?V?Èl???c ?3­?L¸?èÁ?:Ê?oÑ?­×?Ý?Âá?Ðå?Sé?]ì?þî?#x¿¢%¾I?¿óh?#¨ò>ì
U¿åk´=Íi¸=6Æ>z¿HôR½ÒC?u?ïq?ñ/?ÒÀÿ½{¿Ï¾oz?ðC¿(x>ö=ðu¾ôB>n5>fø"¿"{?Ç_G¿·i¾Iv?Ò¶¿v6H¿!.#?ùgS?~É¾6T~¿ý¾þV??¨(t?ØÑ>ù¿C}¿áO¿lóf¾çMÛ>ï`?ôC?áùP?°Öß>¤x¼HÝÞ¾9A¿vq¿?û¿®{q¿rM¿ò#¿¡j¿¾8R¾¶=¨ã>¾ô>¯Y!?Ç@?jY?×¥j?qxv?J}?é? ?ü{?LÅu?ien?¸Òe?Åj\?GzR?A?H?ë=?|¤3?@)?Ñ¯?(?ÿ?õ:?aÁ÷>ºäç>RàØ>¡±Ê>öS½>Á°>lñ¤>þÜ>{>êÂ>úVy>öWh>kwX>{¤I>ÛÎ;>îæ.>ÈÝ">8¥>Í/>Îp>|¸ô=¢Íã=ÕÔ=+_Å=ö´·=»ûª=#=Í=×=èH=Çn=l7^=TÍN=t@=.3=¸¬&=É=$X=S=Èú<§è<jØ<%zÉ<Ô}»<oO?&3u¿j¿_PÔ¾:la¿Rò¿1?Åõ~¿÷øk?k¡½©¿a»%¿/m¾Ma§¾Xõ9¿ûþ}¿3?¾¹}?ÑUT¾ÝÀ$¿u^x?#%~¿õx?L}¿¼ñ{?ÕlE¿I¨=> ?ÇÙ}¿uV>ù\?¿n@E¿b_?ÊVk?¨é½w¿à*¿Nê>,t?%ïT?íº>fg¿mgy¿ÖSg¿!÷¾º=@Þ?9=f?tø?®yf?rî'?[ª>çXE¼ªö©¾õM¿ÁdL¿¨om¿ã}¿û~¿Íyt¿Û`¿TÀF¿¦p(¿Ï¿hµÌ¾&e¾È¾âØ¼^®¬=ºý>>fI>0º>Óá>x3?¹?z}?ß¨+?Áb6?úÐ??H?#VO?3¬U?ª4[?.`?Í<d?)æg?¬k?¼Úm?æBp?Zr?Á*t?¾u?÷w?~Kx?¹Ry?ö6z?Óüz?S¨{?÷<|?Å½|?b-}?}?Ýá}?q*~?Pi~?Æ~?óÎ~?Ï÷~?3?Ú9?fT?dk?N??|?k¬?·?RÁ?¸É?þÐ?K×?ÀÜ?yá?å?é?-ì?Õî?Gõ¹>p¿ú=hy©>N¶¾;s~¿|),?Êû¾.ÊQ?se¿¨I¿Ò¸Í>ÈB?Û??hÝÃ>Üå¾¬ã}¿no>°¿b?¢$g¿ÔLí>ÚçÌ½kó½!Î[½z²>íCA¿ÔÝ?Üe.¿¾­8}?¨öÐ¾+X¿3¬?0è_?ê ¾µÚ¿¿¢ª¾÷3?^£x?·>±ê¿ñ½z¿7èV¿ûó¾êÇ>á[?Í?áU?0Iî>`zi<Þ(Ò¾á<¿~`o¿µü¿^-s¿ÊéO¿$À¿?Ç¾øë¾cX=[>&%ï>c?Yõ>?ø¢W?h¯i?­Üu?]û|?-Ú?;?Ð{?ú2v?¿ên?ýhf?j]?Ç"S?úêH?)>?®N4?Þ/*?Q ?Ä?À?ÙÉ?5Ñø>tæè>ÔÙ>Ë>r,¾>±>t°¥>>Ü">ä_>|z>Hji>wY> J>ÿ­<>·/>Ù#> Z>fØ>ç>0Ýõ=HÞä=Ä	Õ=¤KÆ=(¸=ÀÈ«=üá=Í=ï|=ââ=­åo=<B_=«ÅO=ª[A=Lñ3=ît'=Ö==øô=+.û<¿é<Ù<?lÊ<!_¼<¤n?Cò®¾­Ñ¿¯q¿ø¥w¿~á=v=?ç^¿#¶?ã>Å[¿mj¿8j&¿Ù)¿l¿±Ðd¿Ø8>§g}?Þ©í¾£Ü¾Øb?(·~¿{Ó?¡¿?ño?çá'¿B=0h;?2w¿Fv>Rµi?¯#	¿T¿÷4ø>Ú+s?Ý(
½U]q¿Q 7¿,Îs>Wo?²\?^yN>¿Mv¿¾k¿çr¿¿ =Ç®?ßb?Xù?£pi?ÞÌ,?}µ>O2$<´ ¾â[¿ÏI¿æÑk¿êù|¿H¿~wu¿é[b¿êH¿|*¿aù	¿Ñ¾¨¾ Æ¾è$½µo=Ct8>RP>æ·>F"ß>K ?Ä?ð¤?¿é*?º5?5=??G?ÊäN?&IU?8ÞZ?Ñ¼_?(ûc?­g?üãj?¯m?`p?9r?wt?¥u?«w?9x?µBy?)z?Éðz?å{?ì3|?ðµ|?&}?4}?ÄÜ}?&~?}e~?w~?Ì~?Sõ~??ý7?ÉR?þi?~???¡«?ð¶?»À?5É?Ð?éÖ?kÜ?0á?Qå?äè?ýë?¬î?añz?°ßU¿ <¿].ç¾%?X¿aå0¿Ý¯{?ñ?f¿»´?¿4ÞZ¿Ux"½K¿Ú>gà>¶Bà<t8¿,g¿¾Ø>ù):?ëj{¿·(?ë}¡¾1¸+>ß)x¾¦F?¯GZ¿L~?²Ø¿¬Å¾î?·¾X¡e¿«zñ>cxj?Äj¾¸¿dRÎ¾ÜÑ%?Ã|?NÓ> ê¾â²w¿zl]¿¾% ¾³>yµU?Áÿ?Z?±~ü>EÙ2=LÅ¾³j8¿n(m¿CÞ¿ÍÄt¿²ºR¿O!¿~Ï¾1|,¾-!x=ØË>Gé>©·?|=? ?V?Ë³h?1<u?h¨|?£Ç?b]?N|?\v?&no?¤ýf?·¬]?*ÊS?ÈI?{B??Cø4?üÕ*?õò ?N`?"*?X?­àù>åçé>}ÇÚ>E}Ì>Ç¿>éW²>bo¦>C>Ê>Ðü>¢{>|j>ÀwZ>ºK>=>(0>äa$>>û>üª>à÷=êîå=¯Ö=8Ç=Xm¹=Ã¬=Ú  =0=X"=Ü|=Sq=M`= ¾P=ÌBB=kÈ4=#=(=s=ö²=Z=Zü<Öê<¸Ú<Y^Ë<o@½<åJ>ð°?Ä-¿kd¿	¿9?ý,;>Ëß¾\AD=g U?àÉ¿mÌ¿ug¿)0f¿pç¿ëv1¿ÀDÜ>Y÷g?c¹/¿Há@¾F@?ªîr¿ñ_|?X^x¿p_\?¾À¿"ø½]dR?¹&l¿îy¿<¸0t?¦Sâ¾½a¿RÍ>a.y?#Ñ?=XKj¿·	C¿ À2>¼i?¹c?S>K{ ¿'s¿¾o¿ñ¿@n3;¸U?´^?Á?Ê:l?1?ÊÀ>Cj=xý¾Ø[¿ÈF¿#j¿W|¿¤¿jv¿LÓc¿§lJ¿G,¿¢¿-hÕ¾Lé¾Üì$¾Ö)½/=Ïè1>îU>k5µ>)¸Ü>V ?0Ï?­Ë?ñ)*?£5?ß¨>?G?ûrN?±åT?iZ?"q_?;¹c?¥sg?²j?)m?¯÷o?År?òs?×u?Gñv?x&x?2y? z?±äz?i{?×*|?®|?Æ}?K}?¦×}?!~?¦a~?#~?3É~?Ôò~?ã?6?*Q?h?à|?r?ª?×ª?A¶?#À?²È?Ð?Ö?Ü?æà?å?­è?Íë?î?ä02?ñn½~¿¹Ôt¿, {¿3½Ád?!}¿Þ^?Æw½e¿j=ò¾¼·¡<'«w=C®©¾Ui¿²c9¿jë*?±´?)t¿î&Q?hö¿ûúº>¨+Ø¾+4'?µTm¿4Uv?>ä¾P[¿á³~?ºÖ@¾aqp¿D3Ã>,s?ã®¾ì}¿'Ùð¾Ñ?!p~?ÈXï>Í­Ð¾7ús¿Yjc¿T¶¾->ÔP?Ú?ð^?È9?&=|I¸¾.Õ3¿9Îj¿ë¿ÐAv¿çwU¿íÑ$¿î¾Ö¾ñ<¾==P6>3Öã>^?@;?ÕT?³g? t??Q|?k±?ý{?âc|?ow?ïo?¬g?¬K^?ppT?«?J?í??9¡5?{+?ê!?ªû??¿?øæ?Éïú>
éê>¹ºÛ>ÍbÍ>ôÜ¿>*#³>6.§>ö>*r>¯>jÇ|>·k>Ôw[>ÈqL>*l>>:W1>è#%>áÃ>)>H>&ø=ÿæ=×=$È=Iº=Äb­=·_¡=à0=ÁÇ=Õ=ø"r=ÙWa=U¶Q=î)C=5=Y)=ÇJ=_`=½7=îý<%îë<ÞÛ<sPÌ<¼!¾<ßÍ7¿e?4Yü=b¾Y}<>nu?1Ùå¾>ù>êû¾Äv?ñn½êa¿;ó¿¿hq¿N¢Ó¾0?©>?p[¿º·=?õ][¿xQn?Qh¿OÚA?!ð¿¾?_¾de?æé\¿ìÍ½lk{?Ã¯¾/ªl¿d¡>ÙR}?#>9èa¿ N¿Yäá=CNb?Åi?Å>ë¾Go¿VPs¿7¿§U
½~ªû>(Z?ôP?×n?¾26?ÜûË>·]=è¾7N¿0âC¿êdh¿ñ¥{¿Ð¹¿MQw¿³Ae¿|8L¿ù.¿MB¿ºÙ¾.'¾[-¾¨H½£ì=o[+>AZ>S²>LÚ>4ïý>Ù?³ñ?xi)?&h4?ù>? F?· N?ÓT?>0Z? %_?wc?:g?ój?Xm?ÓÑo?ä÷q?Õs?tu?ÉÛv?Ôx?t"y?z?Øz?â{?·!|?(¦|?ì}?[|}?Ò}?#~?Ê]~?Ë~?NÆ~?Rð~?·?<4?O?-g?§{?c?¿?ª?µ?¿?.È?©Ï?$Ö?ÀÛ?à?Ñä?uè?ë?Yî?:i¾
D?¿_¿V¿Ãe?äeá>9:¿GÉò>o¥â>2Øh¿Ï¢O¿­MÈ¾}Ö¦¾%¿¿Uò¾©ÙZ?Ý>¡ýr¿Iúm?mÇ0¿5?Ô1¿ªÛF?Áåy¿¥ºh?X× ¾¿Ûy?[[½K{x¿ø>£oy?÷m½7{z¿®	¿4	?ë»?I?,¶¾o¿0Þh¿?Ë¾y>1 J?#^?c??Ñ=z#«¾Ä!/¿7Rh¿µA¿=¤w¿(!X¿òF(¿OhÞ¾H|K¾ðÝ=5{>$!Þ>< ?
]9?¥fS?­f?ís?åõ{??i?¼©|?3nw?"op?"h?Hé^?U?£èJ?Ü@?I6?² ,?t4"?¯?T?2u?þû>åéë>·­Ü>$HÎ>ù´À>Kî³>ñì§>ì¨>·>6>¹ì}>Ô l>Úw\>Ê`M>2K?>D'2>çå%>ºx>Ò>å>0Kù=!è=|Ø=É=±%»=Ä/®=¢=â=)m=Î°=As=¦bb=©®R=D=¦v6=Í)==È=Ù=P³þ<°í<Ü<BÍ<	¿<@y¿æ¤$?¤óU?m©ù>ûM?ûnN?ÇÛe¿Y©/?Nca¿,e?!ÉÔ>¾¿ªk¿r¿OC¿_°½Àpa?^Ñ?õv¿Å#¡>µ¼>)9¿G?V?qOO¿ý7!?@^¾JJÕ¾I
s?áÀI¿êEd¾ÇL?|Xv¾è9u¿\ef>N?µyS>@X¿5X¿X§:=LªZ?DDo?P_´>ä®Ô¾Ñèj¿çsv¿]C¿}Ù½(_ì>×GV?Ð§~?Fq?£¸:?Ê×>ô=Ã¾a3	¿ùë@¿ìf¿åz¿Þ¿W-x¿§f¿WýM¿0¿Sa¿Þ¾*b¾ù05¾ì1g½,Qg=6Ì$>S]>ÉÏ¯>Nß×>1Äû>Kâ??T¨(?¾3?~=?ñF?þM?T?µØY?ËØ^?4c?" g?Mj?Ñ,m?Ì«o?ÝÖq?Ö¸s?0[u?3Æv?x?8y?ÿy?WÌz?N~{?|?7|?
}?ev}?WÍ}?©~?êY~?o~?eÃ~?Íí~??X2?åM?Âe?mz?S?Ó?@©?à´?ñ¾?©Ç?6Ï?ÀÕ?jÛ?Qà?ä?=è?më?/î?JIq¿$y?É>Yq¾Ç5é=Ü¦x?¬ïD¾â½U¾¡Â½³ßT?Íý¿ô<|¿HÏ;¿¢)¿²U`¿FÑw¿
±4¾1÷w?ZÌî»dÕV¿Á{}?U¿Ý4?Ö:¿=`?ì¢¿U?±3¾7¿Ip?½õ)=X§}¿ÞB>µ}?'*é<¥iu¿¢Ó¿ïTó>ºó?Å?0ã¾øj¿°Äm¿3à¾Ë<k>ï¬C?~?oøf?Æ?3¼>Ü¾=Q*¿Ä´e¿­Ã~¿ïëx¿4¶Z¿X®+¿æ¾DêZ¾ê`<úòm>ScØ>³?t7?UòQ?¢e?>s?[{?ñy?¥¯?Ýì|?§Òw?µìp?Ý±h?_?¢¹U?®K?é?A?Iñ6?JÅ,?Ô"?^1?¥è?3?ìý>têì>y Ý>J-Ï>×Á>K¹´>«¨>´[>4Á>DÓ>ò>ß²m>Òw]>ÁON>0*@>G÷2>à§&>->£z>*>Ñoú=· é=^Ù=oýÉ=Ú¼=Áü®=mÝ¢=<==ÆJ=>`t=smc=ý¦S=/øD=ÃM7=Â*=n¿=0»=z	=°ßÿ<;î<*Ý<¦4Î<Vä¿< «¾âkk¾²u?ý®w?·U~?/s>8{¿n\z?9g¿3?üJ?*ç.¾Òö-¿º?¿3¨ö¾Ói>£û{?p~>Bþ¿¡8?:>Uú¿P,5?m/¿òÿö>9Z½¿²|?Â3¿	¯¾Ç?lR
¾_V{¿¯>så?x¸>²`M¿¤=a¿Þ¼ù9R?Lt?p0Í>Â½¾f¿K'y¿"%¿WÚ½óÍÜ>«¾Q?9Æ}?/s?}??Ââ>úÈ=ö!o¾·¿$æ=¿Ã¶d¿Þz¿Òõ¿þx¿Xh¿+»O¿Ó2¿§|¿Pâ¾*¾N=¾ní½/ÆJ=5;>*_>Ó­>pÕ>¦ù>ºê?;?æ'?T3?~è<?aE?ÐM?Þ¸S?ÏY?%^?¿ñb?Æf?	j?Û m?o?±µq?
s?1Bu?°v?Lîw?èy?äðy?Àz?®s{?V|?<|? }?gp}?&È}?*~?V~?~?yÀ~?Eë~?W?r0?@L?Vd?1y?B?æ?s¨?.´?W¾?$Ç?ÃÎ?\Õ?Û?à?Oä?è?<ë?î?XJ¿CË>°»i?ÓÜ?2­B?È§e?í?¿ÁÄ>çó¿ír?}+¾Aìu¿Ör¿þf¿¥}¿ >T¿ß·>5Ã?Á¾Í,¿)Í~?ãîn¿<V?ß¬X¿ær?d~¿1s=?£ñ½.M¿äc?¨ >@æ¿öiº=eÉ? !ë=QÀn¿®È'¿=TÓ>S?»?³m~¾ Ûd¿ßr¿1õ¾A>=?÷_}?j?U?F$>uw¾fd%¿@öb¿ä%~¿Âz¿Ï6]¿Ø/¿Äí¾÷Jj¾ìòW;¥`>úÒ>4?­5?¥xP?ûd?Lr?¢2{?¯X?±Ä?B-}?Ê4x?Uhq?@i?n `?\V?Ì7L?4èA?a7?`i-?Jt#?¶Ë?ï|?ú?òþ>¸êí>ýÞ>>Ð>dÂ>+µ>j©>f> h>úo>>×Än>»w^>¬>O>$	A>BÇ3>Òi'>^â>)#>2>lû=H1ê=<ÿÙ=ÛéÊ=Þ¼=¾É¯=F£=èE=ö·=½ä=ß~u=>xd=PT=OßE=à$8=ö]+=Ây=h=â
=	 =Æ4ï<PÞ<¿&Ï<¢ÅÀ<Ñ?,öj¿,ÚÐ>w6Z?r>&?9â¾ÀP*¿=bl?LÂN¿N=tb|?=>6´£¾kà¾ÓU¾#?W}?>b0½;óu¿oá<?©òÅ½Ð·¾^|?QV¿Å£>ÐBå=(-,¿mÜ?ë¿ô>é¾Ú|?»å¼óï~¿Á+'=¦N~?÷Á¸>ÓYA¿	-i¿Vl¬½.I?Øøw?Àqå>&c¦¾f¼`¿Fi{¿×,¿
E¾yûÌ>ÏÿL?b¬|?u?fC?0Ùì>Xæõ=£Z¾× ¿ïÐ:¿Çb¿5y¿¥ÿ¿|Ãy¿yVi¿æqQ¿Ü4¿<¿-æ¾Ï£¾	iE¾@½«8.=¨>¿z>udª>e Ó>i÷>kò?~_?$'? h2?êQ<?OE?.§L?ÈSS?(Y?-?^?¯®b? f?@èi?´Ôl?<_o?_q?s?)u?»v?gÛw?ñx?¯ây?Ä³z?i{?|?7|?-}?bj}?ïÂ}?¦~?R~?¬~?½~?ºè~?#?.?J?çb?ôw?/?ø?¥§?|³?½½?Æ?OÎ?øÔ?¼Ú?»ß?ä?Íç?ë?Úí?b
µ=oÿ¾ûf?Dz?Ã?AÎ>%6¿á.W?Ûñm¿;§e?Àm >÷=¿¯Ð~¿¿wdy¿©e¿v	á>q?,ä¿ï6ï¾Ûq?¢A}¿µ(n?O n¿Þ5}?í1v¿Ì ?×á=ï¶_¿´S?ûÇ>c1¿¼¦?.M>Lf¿MÌ5¿4L²>¢'}?]*?GF¾¡^¿Þu¿¦¿}r>£!6?óÞ{?Qïm?'½?á5B>ö¾\ ¿`¿kh}¿*{¿»¢_¿-S2¿Çõ¾py¾6Ë4¼NS>TÎÌ>cÆ?k3?ùN?Í|c?fÓq?»Êz?À3?Ö?ìj}?x?âq?Ìi?÷¹`?UþV?ûÝL?ºB?Ù>8?ò.?$?·e?ñ??(ÿ>°êî>Dß> ÷Ð><Ã>ëN¶>(ª>Á>ú>¢>®>¾Öo>w_>-P>èA>54>¿+(>(>ªË>8¼>¹ü=ÕAë=ýÚ=DÖË=&º½=¸°=[¤=÷=[]=³~=v=e=¡U=nÆF=üû8=*&,=4 = =D½
=9=PLð<vß<ØÐ<ï¦Á<pÿ~?|Ý]¿ÃÜ¾×+Z>ï0½Õ<j¿ç ½(®
?ß¼¾Á;â¾»s? +?ÏÄ=Ð?|½&g>Y²M?jòe?w©¾Y¿6Wb?Ñ§¾]¾GÊ»>g¹¾çÁ>¶W>2G¿bq~?^æø¾ï¿tv?¢=oý¿ÙU½ôÏz?|Þ>¼<4¿Üùo¿q>¾??³'{?@ý>³¾óZ¿Ñ8}¿æ3¿ ,7¾hì¼>yH?Z{?Ñ{w?-G?÷>Z>F¾ñ.ù¾¬7¿|È`¿Gx¿ü¿}z¿g j¿|!S¿s6¿¨¿¢Òê¾ä ¨¾6M¾~¡½ú¨=&>¾t>·¬§>·Ð>	:õ>_ù
?®?î`&?¼1?Çº;?»D?3L?IîR?ïÏX?ãñ]?Wkb?ÿPf??µi?\¨l?´8o?çrq?bs?ßu?Úv?mÈw?áx?jÔy?e§z?H^{?Ëü{?*|?3ý|?Wd}?²½}?~?-N~?D~?º~?,æ~?ì?,?ñH?xa?µv??	?Ö¦?É²?"½?Æ?ÛÍ?Ô?eÚ?oß?Íã?ç?Úê?°í?ãÌb?9~¿þ%>ûT?£3?·»¾ÝgW¿ÖÞ?t§{¿d?é¦8?*R¿¾;l^¿êr¿®OT¿O=¾òr2?:N?´<¿)Áj¾`W?ûK¿,M|?½ã{¿aâ?ËEg¿W ?7{>aFn¿C@?K®Á>Ö{¿Ý|Þ½M}?.><Õ\¿UÊB¿üe>À&z?&5?
¾JW¿y¿Þn¿Ü9×=ðî.?õz?Áq?ëû%?nµ_>¹j¾9¿]¿\|¿L!|¿Àùa¿5¿lü¾ap¾ÒÄÏ¼ÉìE>÷Æ>ÇS?Ú1?LuM?bb?Üq?ª^z?%?7å?Ù¥}?òx?·Yr?hWj?#Ra?ýW?=M?}6C?°ä8? °.?s²$?`ÿ?¬¤?Ü«?ò ?\êï>Lwà>ÛÑ>}Ä>·>Óæª>s>C·>;©>@>èp>bw`>_Q>ñÆB>!g5>¥í(>îK>'t>:Y>Ýý=^Rì=îúÛ=«ÂÌ=I¾=°c±=ô¥==©=À=¨=¼w=Òf=òV=­G=Ó9=]î,=gî =hÃ=¥^=i²=Úcñ<¢à<ñ
Ñ<;Â<zwí>Iåð½uw¿¨«¿(d6¿çÙu¿U
?BJ½á;>ó¾T¿ñO1?tm?}ý>K¦>É	?»t?¾7?þ'¿þ,¿.y?æ`
¿p±=To->Ã6¾"Eö¼Û>,]¿Ïw?}3»¾ã)¿úl?çB>>!|~¿A¾pu?~?M&¿¤u¿l¢Y¾Iq4?J}?± 
?_m¾»·T¿~¿Bè:¿°Õ[¾~¥¬>îåB?õÐy?/y?K??½®'>·`1¾\ð¾iy4¿¤¹^¿Iw¿ëê¿',{¿ák¿ÜÉT¿Ý^8¿î·¿ãï¾u/¬¾ùU¾¥Þ°½ò.ê<<~
>4»n>ó¤>Î>ýó>ÿ	?+¥?%%?1?#;?§üC?¾K?cR?ôvX?G¤]?·'b?f?i?Ó{l?o?JQq?ßDs?öt?ànv?]µw?Ðx?Æy?øz?S{?vó{?~|?1ö|?D^}?o¸}?~?:J~?Ù~? ·~?ã~?´	?²*?FG?`?uu???¦?²?¼?Å?fÍ?.Ô?Ú?#ß?ã?[ç?©ê?í?s^?Þ0¿>¿ 1>ë~ä<>¨T¿ÄÒ¯¾öíM?ÍÛ;¿e$=)v?¢=_¿+ë?¿+;¿ü+=Ï®b?ã?ætd¿¨À<×0?ôt¿~×?û¿Ú|z?R¿2²¹>²ñÅ>µx¿Ü)?' ÷>aýt¿ PT¾EÃx?v»>@±Q¿°N¿§[>év?	)@?è¨½ÓP¿ù¢{¿Lì¿Ç,=Jw'?Ûw?óÌs?2,?e}>òWO¾Iü¿WøY¿Ñ{¿Ëü|¿¤;d¿E¾8¿aá¿ 
¾"½8>Á>NÜ?/?·ëK?VCa?±Up?oîy?ÞÞ~?²ð?	Þ}?DMy?wÏr?¤àj?ñèa?>X?'N?zÜC?æ9?R/?æP%?±? 8?õ8	?h¡ ?»éð>iá>ï¿Ò>¹êÄ>ä·>
¥«>ü% >{^>ÇE>úÒ>Uúq>wa>'R>É¥C>76>¯)>® > >9ö>ÿ=ãbí=ÁøÜ=¯Í=ir¿=§0²=ÈØ¥=æZ=#¨=²=¼Úx=g=CW=ªH=2ª:=¶-=º¨!=Ïp= =H=c{ò<À¦á<	ýÑ<iÃ<Scý¾ã9?GR¿$|¿æ¿ý¿np?¿Mé-?
o¿#>^?:®N?k)?çlQ?4Æ?>êí>¦L¿Zç¾àí?C9¿Ä>þ½oô><¢ZS¾]?´n¿&l?P[t¾??¿=`?1>âoz¿XÃq¾L9n?ÂÙ?Ù¿]z¿Ä¾j%)?­!??|A<¾N¿}¿V A¿Û¾+>=?x?v²z?1vO?ãA?î=>³¡¾Iéç¾71¿/\¿ý<v¿`Ì¿YÏ{¿wm¿ùjV¿¸D:¿ðÃ¿Ö?ó¾¸Z°¾/¤]¾@*À½	±<Óæ>µh>-9¢>û¦Ë>wÖð>	?öÆ?¶Ø$?_b0?×:?wC?IK?"R?X?YV]?Îãa?þÚe?Ni?Ol?#ën?/q?'s?Ýt?ÍXv?7¢w?í¿x?¯·y?}z?±H{?ê{?óu|?&ï|?*X}?&³}?ú~?BF~?i~?¦´~?á~?x?Ã(?E?^?4t?ï?(?5¥?`±?é»?	Å?ðÌ?ÈÓ?µÙ?×Þ?Iã?"ç?wê?Zí?NY=ÿ¶/>*Æ~¿n=¿Ë/)¿~¿^½>6¦>i¾Òá¾mpz?Èãÿ>ãëT¾\á¾Aðz¾þÆ¾>Êo|?ÓÝ¯>Ô.{¿®>h ?ã»^¿¡x?Áz¿Ü0m?H7¿6 \>®¬?ø~¿GÙ?Þ?_k¿@¡¾ór?:÷ã>Ý0E¿hlY¿²Q>~ÿp?WJ?FEÖ¼H¿¡}¿à!¿«< ½?­Zu?Pv?nø1?£>Ï3¾¦¿±¹V¿ërz¿ú¼}¿3hf¿Ý;¿X¿¾).]½î+>Ý2»>`	?-?é\J?`?éo?zy?ë®~?üø?|~?¦y?@Cs?:hk?_~b?åÜX?ïÊN?±D?y.:?ô/?íî%?©1?KË?ÕÅ	?®'?Íèñ>¤Zâ>¤Ó>ÌÁÅ>d®¸>&c¬>XØ >¡>Eâ>^e>s>Îvb>ãùR>D>á7>^q*>iµ>Å>4	>R >dsî=öÝ=pÎ=NÀ=ý²=¦==M=L=Yùy=a£h=X=Ç{I=L;=Â~.=c"=6=g¡=ÈÞ=íó<åªâ<"ïÒ<ÓJÄ<´£¿í3|?ú/È½N¿æ@¿
Û=R§t?'r¿tJw?=Áe¿ÇET¾ü»]?gz?Àïe?¸1x?m?îA*>lp¿k¶E¾%v?]w]¿°dü>Xës¾o2N>¡À¾µö2?Ïüy¿>[?ÇñÛ½S¿NP?;<È>ãs¿¦¾~9e?ñ@#?$¿C}¿²¬¾x;?é?áÅ?U	¾ôF¿ ñ¿H¿$¾p>8?v?¾|?K7S?Vc?³T>ÆÑ¾%ß¾sç-¿BmZ¿~!u¿b ¿g|¿Fn¿ÄX¿%<¿üË¿fn÷¾´¾µ°e¾sÏ½ÅÁo<ûú=Ý­b>m}>÷0É>{¢î>à	?è? $?´/?ò9?þðB?ÔJ?a»Q?êÃW?]?a?e?ìi?."l?Än? q?
s?Ãt?¢Bv?ûw??¯x?9©y?óz?Ó={?ªà{?Ém|?è|?
R}?×­}?`ý}?FB~?ö}~?©±~?pÞ~?:?Ò&?êC?]?ñr?×?6?c¤?ª°?L»?Ä?zÌ?bÓ?]Ù?Þ?ã?èæ?Eê?/í?eÆO¿¼d?g!8¿Ú½}¿¹¿~¿e1¿0øO?W~¾j¬>)T¿u~D?°T?ó³W>2j½êGð=R)?(}?L·f=È0¿>?F>½µ=¿ùf?fl¿ÂiX?-¿£F={­!?Fê¿v7ë>8+?_¿kË¾aHi?a?Öh7¿ãïb¿U£=ãj?×&S?|Vö<þc?¿E¿Çò)¿Ú­¼ Å?ür?gx?³7?Ox>#¾Ê8¿!\S¿Ì7y¿Åa~¿8h¿í>¿ó	¿G#¾â½>NEµ>!ß?â+?îÈH?æõ^?Ån?y?N{~?þ?1F~?üy?µs?*îk?nc?$zY?`mO?#&E?iÒ:?0?&?IÊ?/^?yR
?Å­?çò>òKã>Ô>¶Æ> x¹>'!­>¡>¶¬>µ~>·÷>£t>nvc>èS>]cE>µÖ7>23+>j>m>-0
>¥ >àï=_ôÞ=ÎÏ=£*Á=Ê³=mV§=4¾=çò=æ=ô{=(®i=áxY=äbJ=fX<=ôF/=]#=Ë=ÈB=÷t=vªô<
¯ã<:áÓ<,Å<¿xÜæ>Ú1?·¾¹9Ê½8?ÔG?ètv¿`t?¿$¿ÖV?Az??b;~?G/@?3¾ô¿»¢=Nw\?)ou¿Òã+?ËÜ¾êyÄ>ÀÀ¿.¬N?N}¿0|F?mëÒ<Ábc¿>?åù>eæj¿eÛÒ¾éZ?à2?ñì¾	@¿Ë¾¾?]â??*?ù²½s?¿>ñ¿ )N¿î£¾-du>{L2?és?¥(}?IÕV?¸p?g$j>`æå½ëLÖ¾8*¿0X¿÷s¿ôf¿Tó|¿%ko¿/Y¿Ûÿ=¿Ð¿yû¾ §¸¾j¹m¾ò¸Þ½Ýú;gí=ý£\>cÀ>¹Æ>mì>ð?{?åM#?E/?³X9?jjB?:^J?ETQ?ÜiW?¹\?%[a? de?çh?õk?æn?ëp?ìr?ã©t?],v?ª{w?x?²y?\uz?é2{?4×{?e|?ùà|?âK}?¨}?Áø}?F>~?~z~?¨®~?×Û~?ù?Þ$?:B?§[?¬q?¾?C?£?ó¯?®º?øÃ?Ì?ûÒ?Ù?=Þ?Äâ?¯æ?ê?í?³n¿e?¹g=³1H¿Ú¤K¿ê¥½sè?E¿ñB?k¿»Á>mt}?3¢?³À¥>°í>_?\Åd?wUq¾O.p¿É}5?Í|Q=.|¿½K?IpU¿PÏ<?×ìå¾fÂ¹½Ê]<?GÄ|¿I±>M@?ÖP¿ðnù¾z^?Ç?p(¿Ò,k¿LÞ;Éc?[?¸°=X<6¿ªÎ¿Zr2¿ ¾½?^o?A{z?Â>=?Á©>Þµø½à³¿%àO¿Ýw¿ë~¿j¿ îA¿òª¿¢¢¾´%©½í>P¯>Y?«)?Î/G?ÁÇ]?öm?Øx?D~?üÿ?'v~?ÃPz?è$t?rrl?¥c?>Z?ÞP?ÍÉE?¶u;?71?³)'?b?Êð?ãÞ
?¬3?	æó>=ä>ÚkÕ>woÇ>»Bº>ß­>Ð<¢>¹S>>>./u>ÿud>6×T>BF>¦8>ÿô+>Ð >ô>"Í
>Ð7>Xð=(òß=)tÐ=½Â=´==¨=Ùo=G=w=6|=í¸j=/qZ= JK=/==&0=®×#=y=(ä='=ÿÁõ</³ä<RÓÔ<kÆ<I¼>ä¾}?}?"!?¥i?6Û¼w#¿_l&?ªú½Q÷l¿	>BN?¹6r?¾b?¦ù>Èå¾ðÉx¿Ó0±>Í4?;ª¿#?Q?-¿¹Y?Ïà,¿¼d?Üñ~¿y\-?æ;">º1p¿ßû(?+?å_¿¡Lý¾'N?È@?FÊ¾þ¿Qªé¾jº??tË3?Àç½R7¿\|¿¥ôS¿µ¾ñxS>Èf,?q?ó~?OZ?Ni?v
>b¼½F`Í¾.'¿¢ãU¿î½r¿ ¿t}¿Yp¿-"[¿Õ?¿ýÏ ¿ùºÿ¾ÞÇ¼¾+¾u¾ûí½û1:­0à=V>>¸@Ä>.6ê>K?6(?"?RW.?Ï¾8?VãA?ãçI?ÂìP?qW?§j\?ea?!(e?ô²h?ÆÇk?un?`Ép?ÜÎr?t? v?Dhw?«x?y?¶hz?ò'{?´Í{?[]|?ÖÙ|?³E}?'£}?ô}?A:~?w~?¤«~?:Ù~?¶ ?é"?@?/Z?fp?¤?N?½¢?<¯?º?nÃ?Ë?Ò?ªØ?ðÝ?â?uæ?áé?Øì?#F¾¨»4>PM?;¼½f'-¾gÖ? «G?³ ¿ÑL}?3Ûe¿ã@Ê½q£s?$Æ^?4)?ßú??°T|?Ã²5?Mæ¿ÁEO¿y$]?V=¾ï6Ä¾$°'?¿´6¿d@?&m¾ü2x¾1S?J!u¿h>,~R?ÂÀ?¿¸i¿Ó¾Q?c
)?r_¿\r¿uô½>¹[?c?xx>÷,¿Ëü¿!:¿°Ê×½#?Såk?"|?øB?å·>ÕôÀ½¼ ¿;FL¿dv¿ëX¿Ýkl¿ßD¿0¿ãª¾r`Æ½Ç>ýT©>Ï?qr'?E?±\?#m?
x?	~?³þ?]£~?¢z?Æt?õl?h6d?3±Z?k¯P?°lF?`<?×1?rÆ'?~ú??k?c¹?3äô>Ñ-å>mOÖ>FÈ>´»>Ø®>ëî¢>«ú>j·>F>§@v>ue>ÎÅU>É G>Fv9>Æ¶,>|Ó >\¾>j>Ê>Ë¤ñ=îïà=`Ñ=ÔâÂ=pdµ=Ô¨=~!=§==h='U}=²Ãk=|i[=1L=>=W×0=ÿ$=j&==V¡=Ùö<S·å<jÅÕ<·îÆ<*{?)û{¿6?ê~?BP|?Ò×N?%6 ¿´½úS>@há>£¿~¿Ô-¾{@ü>¼@?ÎY)?þ¿,>|V4¿I\¿3??÷ ?û{¿ztl?oA¿Q3?K¿.t?Æ]x¿ô®?u>Oy¿p³?)?ÿQ¿_Æ¿Ð@@?ó¶M?w¦¾ô}¿3^¿Ivì>Tg}?¥=?v"<¨H/¿0~¿½lY¿~ãÆ¾2O1>óT&?îên?zÜ~?¥]?cL?Âò>&½m`Ä¾£#¿?S¿vq¿ÔË~¿?é}¿q¿¯¥\¿¤A¿ÖË"¿fì¿åÀ¾×¾}¾Ð:ý½vÎ»R÷Ò=P>B>ÆÁ>äýç>ñ?DG?À!?Ç§-?^$8?Ã[A?qI?ØP?¬´V?s\?\Ñ`?ìd?¤~h?Hk?þMn?§p?	±r?:vt?ÿu?ÈTw?Å|x?t}y?\z?ï{?(Ä{?U|?¬Ò|?}?}?Æ}?tï}?76~?s~?¨~?Ö~?pþ~?ñ ?Ó>?µX?o??Y?è¡?®?o¹?äÂ?Ë?,Ò?QØ?¢Ý?=â?;æ?¯é?­ì?L8?}/¿qy?C'?SJ?a|x?½x>1«^¿¬j?òÅ¿L¿¤8?îá~?oÏe?^Ìo?ÛK|?£6é>@L<¿ïê¿´v?°9Ò¾«â/¾Bqù>`R¿çé>§	¾ÒÆ¾ô°e?$i¿½Ö=üàa?«t,¿w¥&¿û2C?è9?§Q¿¥w¿'¾U¼R?ØÉi?Ø5J>¢t"¿¿ØSB¿º¾iý>h?Z~}?]ÃG?éßÅ>Â½Ðô¾êH¿¸Ìt¿*«¿ An¿ó¿G¿«¿n±¾Âã½_ûé=²R£>þ>E[%?MîC?½\[?óJl?w?Ê}?9ú?ÔÍ~?òz?©þt?	vm?QÆd?K[?OQ?ÊG?eº<?mw2?Ãb(??%?÷?ê>?âõ>bæ>Í2×>}É>Ö»>Z¯>ð £>¡>¯S>|®>Rw>ótf>Y´V>qÿG>F:>x->#!>Áf>>D\>:µò=±íá=ØLÒ=ê¾Ã=^1¶=Ù©=!Ó=ã=Y´=¾s~=uÎl=Èa\=6M=°Ý>=1=PL%=ÐÓ=è&=7=ñ÷<w»æ<·Ö<ÐÇ<2`1?6Ã:¿Òc¾3A?#V?ûFv>ZWx¿ü>\Ï¾S}T?NU¿öR1¿²%¿=:á>a@³>ü-¾©æc¿o-¿¡³H?û>Úli¿2|??_¿:ÁR?ÁÍc¿Ë­}?èèk¿Bâ>ù{Ó>¾~¿Qôð>2=?	UB¿G¡%¿«é0?	PY?¹¾	¿}¿xY¿ÂÐ>õz?PÙE?d×o=Ü¦&¿$6}¿"^¿û×¾
ñ> ?àl?m?²Ö`?B?»É>
XP½9N»¾Î ¿Q¿mp¿*j~¿áR~¿D r¿©!^¿MnC¿Ã$¿oø¿¥þÄ¾¥Ý¾2;¾óY¼°»Å=6zJ>½>øJ¿>1Äå>ä?¥e?Ùø ?¤÷,?b7?²Ó@?ÞùH?P?YV?ïË[?`?¦¯d?Jh?lk?J&n?p?r?;\t?ûèu?6Aw?Ìkx?¼ny??Oz?à{?º{?ÇL|?yË|?A9}?^}?Æê}?)2~? p~?¥~?÷Ó~?(ü~?÷?=?9W?Öm?l?b?¡?Ë­?Ï¸?ZÂ?Ê?ÄÑ?÷×?TÝ?úá? æ?|é?ì?Óýx?^~¿ ¥ì>Hª?@x?
öe?JWÉ¾ÈîÞ¾¾?çÐ½¿é`¿ç½°>dUr?fz?)þ?h_?)Î>ìf¿­pÅ¾õ?î¿7ÁD=Ú>4KÍ¾÷P>Bû<sk¿îzs?ýY¿,¼An??+¿ô69¿¸ö2?o¤G?
Æê¾dÏ{¿ N¾ÜH?,©o?È'>pâ¿3~¿j­I¿KA¾Zë>¯d?³~? ¹L?Ý®Ó>À"½
Gé¾ººD¿gs¿Ñá¿ p¿¸J¿Ç¿2á¸¾Z ¾ÔÎ=øI>n\ù>4?#?FB?ìZ?Tnk?öv?@}?ò?õ~?4?{?hu?Võm?×Te?©ã[?©íQ?°G?Å[=?Ô3?¦þ(?L)?å¦?¼?@Ä?ßö>´ç>ùØ>ÂòÉ>A ¼>°>àR¤>YH>åï>¦@>acx>Wtg>Ù¢W>ÞH>¸;>@:.>Å<">">í£>yî>¥Åó=pëâ=+9Ó=üÄ=Jþ¶=¥Qª=Â=c=JN=T=8Ùm=Z]=PÿM=È´?=¸g2= &=5=HÈ=³Í=ù<¿ç<©×<N±È<;÷m¾Åæ=Óc¿ÅlQ=Rt>§úà¾	`k¿ruf?¡T¿g?ô¾yCp¿¥¾q´=Úõ»:ú¾Ý|¿,»Þ¾&3l?ä<3J¿Y´?cLt¿j?*ßt¿á?¹ÞY¿$(>Í?ô¿L»>ØN?u¹0¿F7¿? ?c?«8¾ëÃz¿ã·¿Fü³>¸w?QN?LÛ=x­¿×e{¿ÍYc¿÷Èè¾BÑØ='³?{i?§Ì?wâc?9Ï#?	 >Ñø¼*²¾¿)¥N¿Eºn¿!û}¿ï°~¿æs¿_¿M2E¿ý¶&¿¿gÉ¾²Ù¾×¾õï¥¼ë}¸=mhD>Á¿>Î¼>ã>&?Z?0 ?êF,?Ûí6?"K@?0H?Ò³O?þU?|[?uF`?
sd?]h?º>k?kþm?æap?ur?Bt?SÒu?-w?ÀZx?ô_y?nBz?Ä{?ò°{?oD|?>Ä|?ý2}?ñ}?æ}?.~?yl~?¢~?RÑ~?Ýù~?ú?d;?¼U?l?N?j?< ?­?.¸?ÎÁ?$Ê?\Ñ?×?Ý?¶á?Åå?Ié?Uì?Ôð¨>Ò ¿'À¾ä}:?ø_?ÙÏ>~^¿Þ>¤î<nþà>ü¿§æÍ½cS;?uNr?mn?÷(?ÿ©"¾Pæ|¿1Sù½Bßx?QH¿Gð> ¹È=ð`¾´Qñ=>G><?&¿G|?}E¿¾¥uw?]# ¿¯îI¿.!?ñÀT?²bÅ¾æ~¿¯º¾ì">?¨t?Ì>Êå¿[ä|¿ûP¿©k¾õTÙ>ì_?ÎU?{{Q?äNá>vóG¼³Ý¾8Ê@¿ÈAq¿Øü¿«¨q¿QM¿Ü¿½5À¾ûå¾J¤³=
;>@.ô>Q!?½@?FÞX?0j?íhv?ZB}?²ç??ý{?zÐu?÷rn?øáe?({\?YR?¤PH?ü=?´µ3?)?,À?\8?8?fI?ÙÜ÷>Åþç>òøØ>ÜÈÊ>Õi½>Õ°>¹¥>ï>>ÅÒ>¡ty>«sh>LX>£¼I>eå;>ôû.>bñ">~·>Ô@>¬>Öô=+éã={%Ô=wÅ=4Ë·=p«=c6=¿-=9è=uX=ùãn=^R^=iæN=ß@=è/3=ðÀ&=.=§i=âc= ú<¾Ãè<°Ø<É<©q¿:]?6m¿Q`/¿}ù¾ôôi¿Þ.ý¾tú|?0ä¿õe?Ó3¼ò³~¿4|.¿¹5¥¾¿kº¾ÈQ@¿Û¿|¿Xâ¾~?¾ôo¾¨¿áÏv?zÄ~¿D¿y?s7~¿©{?¬B¿ü->Þr#?Áa}¿1%>i]?ØY¿ÂãF¿a?6l?Ù½Ov¿k+¿µ>.³s?ã¼U?&>3a¿#y¿ÚÊg¿aGù¾Y=k&?jáe?û?YÈf?m(?S>«>pÕ ¼%ö¨¾è¿ÍL¿Fm¿½~}¿e¿ît¿Ña¿xðF¿/¦(¿®¿L&Í¾Ó¾Ýp¾^ß¼)>«=ÍT>>ü>ÛOº> Lá>¶?d ?£g?+?ÊQ6?Â??
H?¶JO?6¢U?ó+[? `?/6d?fàg?ªk?aÖm??p?ÎVr?ç't?»u?Ñw?¢Ix?Qy?5z?ûz?F§{?<|?û¼|?²,}?~}?Yá}?ÿ)~?îh~?q~?©Î~?÷~?ü?ª9?=T?@k?.?q?d?W¬?·?BÁ?«É?óÐ?A×?·Ü?rá?å?é?(ì?b¶¿7ìÈ>´s¿÷&(<5!>sf¾ÏÄ}¿|0?¿¿q\T?ðc¿¦¿gþÆ><L@?pk=? f¾>Eê¾}¿lØ>¶a?Wh¿°ð>°Û½Óø¼Äyt½}+µ>Æ#B¿Wç?]-¿3¾ç]}?BÏ¾ç¡X¿< ?FE`?e·¾à¿.Ç«¾¦2?9Ãx?ñ·>d¿¨z¿æW¿¨¾.úÆ>ÕèZ?oÐ?·V?ù¼î>x²x<ÁÑ¾ö½<¿Oo¿<ü¿¡:s¿Ù P¿Ý¿~Ç¾æi¾Ýk=&&>½÷î>ªø?æ>?ÐW?§i?¯×u?Ïø|?¥Ù?¶<?lÒ{?g6v?íîn?µmf?]?(S?aðH?>?T4?5*?±V ?É?x?[Î?ÈÙø>îè>¶ÛÙ>ÍË>G3¾>ô±>|¶¥>À>'(>Ød>Ïz>ñri>³Y>,J>
µ<>¡½/>ù¥#>Ö_>¹Ý>Ý>næõ=ãæä=ÈÕ=SÆ=¸=8Ï«=è=Ó='=¿ç=¹îo=¨J_=ÍO=öbA=ø3=@{'= Ü==ú=¦7û<âÇé<ÇÙ<äsÊ<q¦I¿çvk?£¼ ¾ü¿z¥s¿/v¿ãæ>Þä9?kv\¿Õö?Ñê>3ºY¿¦ßk¿eý(¿¯5,¿¤m¿
c¿Z>b}?;ñ¾ÂZØ¾óa?å~¿Ãá?)¿po?ÙÞ&¿µà<¬1<?Êév¿)~>oj?¿g¿¦U¿èãö>cs?e}ÿ¼t)q¿*e7¿Äq>èn?§Ð\?x!P>ñÆ
¿únv¿àk¿Z¸¿­=t?ewb?sø?ßi?Àó,?DÙµ> /<û±¾Ä;¿!I¿Äk¿õ|¿?J¿Qu¿ågb¿Á¨H¿*¿Î

¿A4Ñ¾Ë¾¾ ½ü=f?8>F8>TÐ·>Éß>?Å¼??³ã*?/µ5?8??~G?4áN?FU?|ÛZ?oº_?ùc?8«g?jâj?-®m?0p?y8r?t?¹¤u?ÿw?p8x?3By?£(z?hðz?{?£3|?°µ|?a&}?}?Ü}?ã%~?_e~?\~?ýË~??õ~?û?î7?¼R?ói?~?w??«?ê¶?¶À?1É?Ð?æÖ?hÜ?-á?Oå?ãè?üë?)å~¿¾Øx?U²Z¿RÚ6¿w#Ù¾*ET¿ù¸5¿ëuz?9¡c¿c?-3¿bX¿ðw¸¼â>ïÜæ>=¹(=56¿xfh¿.hÓ>Çù;?Õðz¿Hâ&?\m¾¿Ý#>Aøp¾Ê ?pY¿H~?Xü¿IÃ¾-æ?OÙ¾ +e¿É/ó>Vj?Än¾(Á¿ýÌ¾U&?qõ{?Ò>lë¾Ów¿¾/]¿P¾ùP´>ëU?rÿ?)]Z?#öû>*J.=zÉÅ¾8¿>m¿ÿß¿Ûµt¿ÃR¿K-!¿a»Î¾å+¾þXz=>¹é>PÎ?s/=?LV?j½h?[Bu?«|?hÈ?*\?|?Vv?6io?øf?¬¦]?ÙÃS?UI?<??Úñ4?´Ï*?Ûì ?kZ?|$?S?hÖù>'Þé>G¾Ú>tÌ>ü¾>6P²>(h¦>X<>2Ä>àö>é{>'rj>nZ>­yK>§=>G0>Z$>*>z>
¥>Ìöö=äå=þÕ='/Ç=e¹= ¬=  =vx==w=yùp=ñB`=´P=:B=FÀ4=5(=e=f¬=>=-Oü<Ìê<ÞÚ</UË<0¾=Ã`p>?Ç*3¿sÖg¿d¿:P4?ÞS>JEê¾§=%R?Í¿bï¿¯e¿d¿`È¿ÎÒ3¿3¸Ö>(i?åÈ-¿´J¾Ò B?/s¿n³|?Ïx¿¯>]?&¿Äì½Q?°¥l¿óå<Ús?{1ä¾¬Ga¿í+Ï>Çüx?mU3=j¿eB¿;5>\i?§Ic?;X>¾ã ¿­Js¿@o¿·¿F;ª?+Û^?¥Ä? l?ÿ`1?]À>]ýÿ<ã^¾Ý¿UäF¿44j¿^|¿z¿av¿?Åc¿[J¿w,¿Ý
¿2>Õ¾!À¾í$¾­(½=¹=F(2>Ñr>Oµ>ÏÜ>Ì ?}Ø?êÓ?71*?
5?®>?|G?LwN?wéT?´Z?t_?»»c?Óug?ø³j?Îm?ùo?r?!ós?Æu?òv?,'x?:3y?§z?'åz?Ï{?/+|?]®|? }?}?Ø×}?Ã!~?Ëa~?C~?OÉ~?ìò~?ø?06?9Q?¤h?ì|?|?³?ßª?G¶?)À?·È? Ð?Ö?Ü?éà?å?¯è?Ïë?ötë¾,ÙD?-T!¾Eá¿Wum¿M~¿»Ì¾3l?¿ûf?Wðö½{o~¿EÜ¾¦þ=ÆÃÑ=3¡¾Fe¿gE?¿Ñ$?õ
?±Ð¿ATM?wý¾Qo°>9Î¾£y#?úk¿Qsw?] ë¾®]ü¾Ô?HM¾ëio¿aDÈ>B0r?>¾/2~¿4í¾[?0<~?Jhì>ñ~Ó¾ªht¿UÌb¿*¯³¾±`¡>/¤P?Èâ?¶z^?¹{?!#=°¹¾T4¿uk¿"¨¿4v¿ -U¿-r$¿ëÕ¾$X:¾ÞÎC=pë>lrä>S?s;?üT?ÑÎg?ò¨t?ÌZ|?û³?Üx?:\|?Füv?Ñáo?úg?¯:^?§^T?|-J?ÉÚ?? 5?Úi+?ª!?ë?C¯?²×?¸Òú>wÍê>£ Û>/JÍ>ÄÅ¿>]³>¾§>Þâ>/`>Ü>ñ§|>Mqk>\\[>#XL><T>>ç@1>%>z°>w>57>%ø=Hâæ=ZêÖ=1È=æ1º=ÆL­==K¡=Ï=¶=P=7r=9;a=±Q="C=u5=Þï(=É6=ÄM=l&=´fý<(Ðë<õqÛ<y6Ì<×Rc?¢ª#¿Í|?Öú<kN¿¾-ûÕ=¡}?*qÅ¾Â#²=à¾Ü!~?%â½'g¿«u¿U§~¿©«t¿¦|ä¾¨&*?âC?¿W¿Ã=Xä?ôm^¿êQp?;j¿ÞE?UÈ¾Ã6¾Gc?ß½^¿ÍÆ²½Íz?ÈKµ¾£k¿{æ¥><ø|?À³ò=`Þb¿ûL¿Vð=«c?[#i?ÔQ>yí¾¢·o¿ór¿W¿2ô¼7Jý>[?½_?n?´´5?ÕÉÊ>T=½ý¾.¾¿2D¿h¿¼¹{¿µ¿9w¿Òe¿|L¿®Y.¿Ó¿DÙ¾Ñ²¾õ0,¾	;E½\t=|,>?¬>oÍ²>Ú>¨*þ>ó?	?'~)?]z4?ú#>?F?ÿN?T?9Z?K-_?"~c?6@g?Vj?E]m?åÕo?mûq?Øs?»vu?Þv?Õx?1$y?z?ÚÙz?{?²"|?§|?¨}?ÿ|}?Ó}?~?4^~?'~?Æ~?ð~?ò?p4?µO?Tg?É{??Ø?"ª?¤µ?¿?<È?µÏ?.Ö?ÉÛ?¤à?Øä?{è?¢ë?ÁZÿ>oZ½o&?þ+¿Íôn¿{å1¿¶ù>G
?âK¿±'?rH·> q¿îýB¿²ª¤¾|å¾¡¿WC}¿²¿¦S?>_v¿ãÐi?m)¿°A?¿È±A?*x¿ok?÷­¾ZÈ¿Ø¿z?C Á½rDw¿T>tx?lA½q5{¿"¿»?V?Ç?:Ñº¾nip¿·ñg¿p¼Ç¾Í0>!K?{z?P_b?ß
?²Ç=Ãx­¾ø/¿Åh¿¬T¿gw¿;ªW¿«'¿ÓÝ¾óÀH¾Ó;=6}>ù#ß>Äk?Ë²9?à§S?ÅÛf?yt?X|?]?Ì?|?5\w?¾Xp?h?Í^?}øT?ØÊJ?èx@?Ý+6?,?"?Q{?Í9?\?·Îû>¼ë>ËÜ> Î>ÐÀ>hÊ³>>Ë§>S>ü>Ë>æ¸}>dpl>J\>6M>É#?>2>Ã%>ÅX>Q´>]É>zù=õßç=Ö×=8çÈ=Éþº=®=Ùü¡=(Ã=îO==ôs=3b=ÇR=7èC=£P6=,ª)=.ä=#ï=¼=;~þ<JÔì<dÜ<ÄÍ<vã]?¢¿­B?>?Ã±·>Ü8?}_?9qW¿ñÏ?;T¿	o?%ù«>¸ß%¿fr¿Ó9w¿:M¿V¾UPZ??|hs¿X>*{Ð>íé?¿0[?]CT¿c'?Sb{¾áÈ¾íp?iM¿_TN¾WÜ~?Z¾®ãs¿Åv>bO?È/E>øZ¿¤V¿bi=\?¨Yn?úî¯>ß®Ø¾{·k¿îu¿ÌÛ¿Ê©½Dï>]W?ÐÉ~?ÑÛp?Bî9?ØÕ>»ý=i¾î	¿sA¿ëèf¿9{¿Ù¿Gx¿hf¿Ó­M¿P70¿¢¿¹EÝ¾ £¾Á3¾Æa½"\l=õ%>ä>J°>0MØ>a&ü>÷?´=?Ê(?&Ü3?÷=?$%F?L¢M?M/T?3èY?Næ^?K@c?a
g?Vj?4m?²o?¶Üq?ê½s?_u?Êv?kx?y?z?Îz?-{?+|?|?B}?sw}?AÎ}?t~?Z~?~?éÃ~??î~?ë?®2?0N?f?¥z??ý?d©?ÿ´?¿?ÁÇ?KÏ?ÒÕ?yÛ?^à?ä?Gè?uë?s²?Å+U¿¸ÿ?PU=Ìà¾]/½j?ó	½7å³¾L}=DØ@?ô$3¿±Íu¿6Æ(¿¢Z¿Ò[T¿ºP|¿B¾Ùµr?Æv=O#_¿öz?æÿL¿{+?f2¿&Z?vÞ~¿gÏZ?qýV¾U1¿Í"s?!TV<¦|¿;[>$Ü|?µTê;;Ïv¿¿û>¡ÿ?(ñ? ¡¾GØk¿.l¿ÍnÛ¾¸u>ÁBE?ªÆ~?÷	f?#?Âþ="$¡¾F+¿¡\f¿¦å~¿Ãx¿aZ¿#Ù*¿¢$ä¾5W¾´D­<7q>éÍÙ>´3?Oí7?yNR?Käe?ñis?B®{??ù©?Ü|?%ºw?ûÍp? h?3__?\U?ggK?_A?È6?Ó,?4­"?S?Ä?Dà?gÊü>U«ì>¾dÝ>çôÎ>¸WÁ>V´>§|¨>´/>ü>¯¬>ÇÉ~>lom>Ò8]>ðN>Mó?>Ä2>"x&>>(Q>[>Ê'ú=Ýè=àÂØ==ÃÉ=©Ë»=MÊ®=s®¢=h=Úé=à$=°t=Ç+c=ÝiS=K¿D=Ð7={d*===ÈR	=Áÿ<lØí<"VÝ<ùÍ<ì3G=÷À¿×¹?»~X?Ôf?b]?ÜÿÌ>ÒÚ¿­o?ó^¿](?à6?¼¾­|@¿Û1O¿Â÷¿->µw?:Ò¢>¿?öú>V"J>aU¿o>?Ï7¿1?²LÀ½Câ¿SKz?ë8¿áB ¾dú?%¾qz¿ä>Sþ?úô>Ø)P¿,_¿vÃ\»C`T?çèr?X!Ç>DrÃ¾	Lg¿x¿Q+#¿(È½©à>áR?û~?ýr?>?KUß>:ç½=)t¾²¿¦>¿2.e¿Iz¿Wñ¿ÀËx¿w®g¿NO¿p2¿@ø¿&Cá¾¾zO;¾<O~½þÌQ=#Ù>Ý>Å­>
Ö>Ç ú>»'?¬q?J(?h=3?x=?ÏªE?47M?°ÑS?zY?^?5c?VÔf?'j?²m?o?ß½q?#£s?[Hu?Üµv?ïòw?íy?aôy?Ãz?Mv{?|?3|?Ô}?áq}?nÉ}?F~?úV~?ä~?1Á~?åë~?á?ê0?¨L?°d?y??!?¥¨?Z´?}¾?EÇ?ßÎ?uÕ?(Û?à?_ä?è?Hë?S¡?2q¿ðJ%?àD?S>ÂF?HÑx?¨w¿¥ÄN>¦*à¾Ç6y?ÿ¡¾Z}¿e¿nV¿\Þw¿|Xb¿Ãè=Ê®?÷ F¾Hj;¿.ð?õg¿DvL?P¿fm?<¿E?_ï½PF¿¨Hh?+Fö=N¿Ã«ú=¡^?­s³=âq¿m)#¿÷Ý>¬z?§?¥þ¾û·f¿HÌp¿H½î¾
aN>,??Ç}?»yi?ØG?Ã&>¸´¾Kõ&¿~×c¿[~¿½¼y¿àn\¿Ãú-¿,ë¾'re¾À <_Ùd>hpÔ>2÷?#6?iðP?fèd?]Är?R{?c?d¾?;}?x?Aq?Ui?²ï_?A)V?)L?-³A?¸c7?¦5-?ïA#??+N?Cd?ÆÅý>âí>{FÞ>ÊÏ>~ Â>)Dµ>ú-©>Ö>Ì3>>>Ú>dnn>û&^>HóN>ÊÂ@> 3>,'>P©>ûí>¤í>8û=EÛé=¯Ù=@Ê=¼=¯=`£=×=Ä=&´=l$u=$d=óPT=_E=þà7=È+=õ>=à1=õè	=£V =Üî<8HÞ<YÚÎ<PoP¿¯¬>|C¿Ì¡#?ØÞt?F@O?0Üp¾¥M¿ºz?Ì(f¿É5j>Ùr?a>¿â¾åÖ¿¨¾2ï>
Î?îØK=){¿c.?ú³¼¬Ø¾·?Ø%¿ê¸¾>²pu=4Å"¿Û7?ý!¿9×¾p$~?h}½>~¿è¡=î?n¬>RDE¿íÊf¿ö½L?æÍv?ØÚÝ>ïÎ­¾Pwb¿1¿z¿%C*¿³¾s Ò>èN?b}?îöt?yB?æqé>9¼ç=}a¾,¿¡Ë;¿ec¿¥}y¿ÿý¿jy¿rìh¿8èP¿å3¿¡ë¿A<å¾{¢¾¿ÚB¾aj½;7=±»>.¤|>»?«>ÅÓ>Ýø>Û@?¥?a'?"2?~<?
0E?¸ËL?¹sS?qDY?W^?àÃb?f?Løi?¨âl?\ko?çq?As?1u?¡v?`áw?³öx?-çy?©·z?al{?	|?À|?_}?Il}?Ä}?~?WS~?¾~?w¾~?é~?Õ?$/?K?[c?Xx??D?æ§?´³?î½?ÉÆ?tÎ?Õ?ØÚ?Óß?#ä?ßç?ë?Y,¾¾î¾>'¾rb~?v]?nQx?ÐÂ!?>Or¿\i1?T¿Õ¤w?Ùü>ÞX¿Üp¿;z¿¿ìí1¿È§>éy?×}Þ¾Ü3¿-x?É+y¿Ûÿe?«*g¿ô	z?Ä)z¿¼Q,?Kæa=Y¿lVZ?=ug>Ð¿Øò<.÷?ò+>¸ái¿ù0¿X ¿>õ~?Êß%?Ëú[¾¥a¿Ñ|t¿Ï ¿ß&>Ô8?d}|?¼­l?¤J?µÎ6>,¾FO"¿ó5a¿µ}¿^Äz¿¶^¿11¿G&ò¾	¹s¾|Ï´»-rX>¨Ï>Q¶?DT4?·O?èc?Àr?:óz?hB?Ð?S}?þox?d³q? i?`?-ÀV?L?ROB?Õþ7?Î-?MÖ#?w*?þ×?è?ÔÀþ>-î>(ß>ôÐ>!éÂ>à ¶>5ß©>A|>Ï>SÐ>¨u>Mmo>_>ÑO>>A>&G4>á'>Q>Ê>Ä>]Hü=çØê=[Ú=@{Ë=ee½=ÎG°=¥¤=,³=®=mC=&/v=Re=8U=smF=+©8=Ù+=Yì==Ó="
=fâ =±àï<N:ß<£»Ï<¯m¿£Ôt?|¿^¶å½¿L ?Røx>ÍjF¿1¥¾~8?X¿¿¾Ì}?E?ÐH½+X¾Í)²=³8?ìèq?9c¾ºf¿XU?©	v¾ðj¾}Òà>FûÛ¾W±[>îjY>ìQ=¿A?9Ä¿®¨¿ç_y?Ö=3ã¿ác¼Óa|?/Ð>úh9¿~m¿ÉÃý½ßýB?ëz?±ô>>Ð¾;]¿õ|¿Ú 1¿	þ(¾"Ã>úI?7ã{?Çv?÷÷E?hqó>¼>^üM¾wü¾ä8¿!a¿ª¤x¿ÿ¿?7z¿y"j¿-|R¿÷´5¿»Û¿ö0é¾Üc¦¾6cJ¾a«½3¨=Ï>w>½¸¨>ÆÑ>¤ö>WY?Ç×?!¬&?Tþ1?õ;?Õ´D?×_L?iS?òX?®^?Mb?gf?èÈi?u¹l?Go?Îq?Ams?u?Jv?¾Ïw?içx?ëÙy?*¬z?kb{?` |?E|?äÿ|?«f}?·¿}?Ü~?°O~?~?¹»~?(ç~?Ç?\-?I?b?0w??f?%§?³?^½?LÆ?Î?ºÔ?Ú?ß?æã?ªç?íê?a{¿Xª?²x[¿r?òy?ÝCf?ª=Sv¿Ö}x?ËZ¿á<?*;?g1¿°}r¿¶}¿·i¿g¼ß¾ÃR?;ë`?7Æ%¿f,®¾Òe?é¿×)w?Múv¿¥«?Ýn¿?ê?>@+h¿¦{I?ç¨>v}¿2Æ½¨¤~?A|>ma¿=¿væ>Ö¥{?F0?^'¾¹ÖZ¿Ø¬w¿Ó
¿n6þ==2?èz?'¥o?*#?SR>w¾å¿Vx^¿¹ó|¿´{¿%ì`¿54¿Bù¾ù¾o¼sL>ÖÉ>"q?Ë2?j&N?tãb?mq?Jz??òÞ?i}?çÇx?#r?j?(a?VW?B8M?ÍêB?h8?øe.?Oj$?¹?a?ªk?»ÿ>6vï>W	à>¹sÑ>¡±Ã>z½¶>Zª>l">@k>b>üý>&lp>&`>Ù¯P>ªaB>¦5>(>Éù>'>á> Xý=Öë=Û=>WÌ=@2¾=±=;Ã¤=X=·=²Ò=ß9w=f=V=DG=Wq9=c,=¼ =t=O=)n=Òäð<d,à<íÐ<b£A¾<N?ìÉ¿µfK¿Ygd¾\»ß¾ÊØ¿mj>p/v>sS½g7-¿U?ÖàT?¤>v>rÔ>Cf?7O?ô¾C¿e»p?Òä¾#Ö¼2\>µº¾õÀO=5*¸>ýS¿½v{?Ø·×¾
í¿Ýºq?wë>K|¿cÒ½ax?/¥ò>¦,¿2s¿:<¾O\9?´|?5Ö?Â¾ W¿*~¿Â7¿l5K¾ª´>`CE?±z?/ox?öÂI?Rý>U>ýÇ:¾¿ô¾ï5¿ÿª_¿¾w¿Zô¿6Þz¿Pk¿é	T¿F7¿È¿0!í¾^Iª¾ÂèQ¾oê©½ =|>wq>0¦>½8Ï> ô>1q
?ê	?2ö%? ^1?h;?19D?óK?¾¶R?oX?Ç]?{Fb?è0f?Si?l?#o?`q?&Rs?u?àxv?	¾w?Øx?Ìy?  z?iX{?µ÷{?Á|?aù|?a}?Ôº}?¡~?L~?g~?ù¸~?Æä~?¶
?+?H?®`?v???d¦?f²?Í¼?ÎÅ?Í?\Ô?5Ú?Fß?©ã?uç?¿ê?0¿Äþ?Br¿Z¾±?Ñ>8¨¿#¿i
s?Bf¿«>Át\?Ëi*¾­@¿óu^¿8]8¿ÇÙ
¾£öK?£y7?Q¿¬Åà½jF?oæ{¿AT?Îÿ~¿Q~?Aê]¿ß>%ù¡>ã»s¿Ðñ5?aoÛ>ÈÇx¿x¾i{?~©¥>ô³W¿H¿ >Yx?Ûµ:?Lä½ùT¿±Zz¿Gý¿ J®=#i+?©	y?:_r?Tæ(?°m> µ]¾Õ½¿ý[¿|¿5|¿c¿7¿íÿ¾Ï¾Ò½®?>"-Ä>µ'?Á¨0?ºL?oÚa?v»p?¿)z?ö~?ë?òÀ}?Ìy?r?ôj?a?ëW?ÑM?C?n39?uý.?òý$?kH?êê?ï?þZ ?ýcð>uêà>SHÒ>þyÄ>øy·>hA«>È>ä>Çó>E>ïjq>)ñ`>Q>1C>Ê5>öI)>ÿ¡>_Ä>û£>Þhþ=!Ôì=ÊsÜ=:3Í=ÿ¾=HÅ±=Ñt¥=Õý=Q=÷a=Dx=Ùg=.W=H=9:=°M-=G!=ù=|«=ëù=ôèñ<yá<6~Ñ<Ì_9? É;£>ÿ|¿FÞS¿¢¬i¿ Q¿HRE?Ö ¾°Àß>þIq¿"?n|?ù(?ñZý>1?¸¢}?µ?2¿ý¿t~?¯Ã!¿£6>í+=Hé´½Fxê½%Cÿ>¹Of¿'Úr?í¾ß4¿æKg?øxq>lß|¿çü@¾±:r?ÑÞ	?¿õÞw¿Ùvx¾»$/?yf~?ðT?gÞU¾\Q¿?¿$>¿½0m¾4Ó¤>`@?y?íy?épM??I2>&'¾xì¾3î2¿J¹]¿Ëv¿Þ¿I{{¿vl¿`U¿ãF9¿í±¿Ýñ¾,®¾JkY¾`'¸½JùÎ<÷Z>vÝk>9§£>sðÌ>Týñ>j	?t;?²?%?&½0?ªÚ:?½C?çK?ºWR?vLX?6]?jb?ÿùe?ii?fl?ÿn?<Aq?î6s?qêt?adv?A¬w?¢Èx?<¿y?	z?^N{? ï{?6z|?×ò|?][}?ìµ}?`~?WH~?6~?5¶~?aâ~?¤?Ç)?zF?U_?Üt??§?¢¥?¾±?;¼?PÅ?.Í?ýÓ?ãÙ?ÿÞ?kã?@ç?ê?_r>\/?±½½¾ÛQ¿æ¾ì¾9t¿Ç°½Ú¼"?Q¿¯I¾ã-?Ðæ>e#ã¾Ï"¿p=å¾N7>tp?Å[ ?Páo¿uÝ>â1?pMm¿ü2~?jú~¿¼v?^¶G¿µw>ôá>è{¿û?cä?Âq¿ZÞ|¾/Lv?9Ì>ÄL¿ª2S¿ç?>#%t?SBD?B¨s½{âL¿ó|¿Ó±¿=4<=»Z$?áv?CÛt?¦|.?so>)D¾ÇÓ¿CªX¿ {¿BN}¿ e¿(:¿º\¿ê¾pf:½\3>»³¾>Ú
?1Ì.?$JK?Í`?Îp?¿y?ÏË~?sô?ô}?­qy?Ëþr?ük?â%b?X?jN?ÄD?éÌ9?/?9%?óÖ?t?Hr	?
Ø ?Qñ>]Ëá>ÂÓ>8BÅ>Y6¸>_ò«>n >y¢>o>>¨ir>ßa>AlR>i D>6>]þ)>0J>%a>6	>yÿ=¸Ñí=ý_Ý=3Î=ðË¿=²=e&¦=(£=gë=;ñ=NOy=h=AíW=ªòH=®;=ü.=ô!=V·=©A=®=íò<â<_Ò<Ì¹x?%\L¿«Åm?Ø ¿ù|¿<v¿P¾f?îE¿ùS?ÌC}¿TÝ£=Áw?
ne?¡ÄE?èd?°Ú{?üÌ±>Â~]¿%Ð²¾Öî}?EI¿À>ç}ò½÷Ì¶=D¾Â( ?®és¿¥íe?>¾ÚG¿Ó1Z?iÎ©>^x¿î¾Æj?¨¤?R­¿i{¿¢ ¾g_$?ñ?­|?áH(¾Q6K¿Èº¿=FD¿ùr¾j>øR;?¨Iw?FB{?IQ?wZ?ôF>§,¾H[ä¾\à/¿º[¿|Ëu¿¼¿r|¿pm¿W¿Á;¿î¿éóô¾Ô²¾°ê`¾bÆ½Ê<8>Bf>¼¡>ê¦Ê>Bñï>?dl?¢$?Æ0?ÃL:?@C?ÚK?\øQ?-ùW?6]?Èa?àÂe?9i?Ú<l?HÛn?Â!q?s?¹Òt?ÌOv?gw?'¹x?Ï±y?ez?GD{?Bæ{?¢r|?Gì|?­U}?þ°}? ~?¤D~?~?o³~?úß~??ù'?êD?û]?°s?}?Æ?à¤?±?©»?ÒÄ?ÀÌ?Ó?Ù?¸Þ?.ã?ç?cê?²r?«¾(Lï>_0{¿¸#I¿°áS¿ëq¿Ôtý>B>)½¿/dn?n6 ?î½k¢¾"þ½dÝó>?´~>Ð«~¿|l¶>Ü>¥»T¿Ðs?Mêv¿«ýf?yÈ,¿j(>½ ?¿»ã?¦?üÓg¿ý«¾@Vo?ÌAñ>j®@¿Å\¿ßú=o?õ/M?1©ë»¢+E¿{*~¿$¿UHÜ;:?ot?w?Jì3?tí>q}*¾sÔ¿U¿z¿ ÷}¿g¿¬ç<¿êº¿~&¾Gòp½û&>Ò3¹>g?(ë,?;ÕI?j»_?*Lo?ÛQy?ì~?û?å$~?Ãy?Üis?k?w°b?Y?ÔO??¹D?×e:?+0?!$&?.e?Þü?Kõ	?íT?Ä>ò>¬â>ñÓ>N
Æ>ò¸>?£¬>|¡>þ=>
>¼>Qhs>Íb>fJS>¼ÏD>ûL7>¿²*>^ò>æý>&È	>§D >LÏî=-LÞ=*ëÎ=ÅÀ=½B³=ø×¦=yH=M=~=Zz=]ýh=SÔX=»ÉI=ÙÉ;=HÂ.=ä¡"=³X=Õ×=p=6ñó<¤ã<Ê@Ó<ÏÌ¦>æºu¿Qb?E>ø\¿¬¿¬>n^?Êó}¿V?\N¿wº¾f«G?øk?Çr?|~?¼a?&-w=Lôw¿¯fÐ½3o?g¿i?í¾Ñ/>
¸Ü¾å<?|¿÷ëT?+½ÎôX¿VJ?þ-Ù>Åq¿³µ¾NÌa?k(?N¿t~¿ê5·¾
?Nþ?BG#?Îµô½ÆwD¿þ¿Û$J¿%¾¦Û>ì6?Æau?Gm|?sT?Ü?[>MÉ ¾þ+Ü¾0Æ,¿­Y¿¾t¿¿¬|¿Bªn¿SX¿ÖÅ<¿}z¿AÖø¾¦èµ¾Ùfh¾0Ô½{2I<(ö=±¤`>>&\È>îãí>û´?º?Ñ#?áy/?c¾9?¨ÃB?h¬J?¥Q?¥W?£í\?a?e?s	i?ýl?æ¶n?(q?) s?èºt?";v?zw?©x?T¤y?¶}z?&:{?zÝ{?k|?¯å|?÷O}?¬}?Òû}?í@~?Ë|~?¦°~?Ý~?x?*&?YC?\?r?y?ä?¤?l°?»?SÄ?RÌ??Ó??Ù?pÞ?ðâ?Õæ?5ê?@óH?òo¿©éy?3P
¿Ê¿«~¿Ç¿¿ªf?=Ø¾®Vß>ýsf¿0{-?bçc?|£>B-=1áZ>Eã9?y?·	¸¼{}¿.Ú?Ï>d>_93¿Ë`?g¿÷§Q?¼Ã¿³<ð,)?9¿ùÛ>£X1?9Ô[¿âØ¾#f?Ãi
?3¿UEe¿ýâl=i?wU?Ò8=ý<¿kJ¿zB,¿-½?Äµq?¸y?49?rO>µ¾À	¿*pR¿Þx¿@~¿Ø
i¿ò¼?¿í
¿-&¾¹½	ø>­³>©2?±+?Ø[H?s¥^?n?àx?Ül~?çþ?NS~?_z?8Ós?Âl?Û9c?¤Y?ºO?RE?7þ:?<Á0?«¶&?ó?{?x
?§Ñ?Ã+ó>ã>ÅÔ>AÒÆ>Æ®¹>T­>\º¡>tÙ>¨>è>ëft>äºc>(T>E>_8>g+>>¤>8Z
>¿Ì >ÜÌï=Z8ß=ÇÏ=eÁ=u´=§=Êí=3=Á=¹d{=õi=d»Y=Ì J=<=|/=FO#=ú=n=2=Võô<¹ôã<"Ô<t¿Òs²¾9ú]>QkW?¿£©=éuÐ=RV?ÝÞ>5h¿K\f?ÑôÞ¾A<¿Ð3é>0r?/Å?]z?,0?oSm¾vï¿ô>èS?Cy¿Ì6?¯Ûõ¾àhÜ>ç¿î)U?Cð¿j"@?3{=2+g¿8?1?h¿kÞ¾æZW?¥6?Çã¾O¿ú¿Ó¾ÉN?@½?¼®,?;Y½Î_=¿ZÝ¿z¾O¿ «¨¾÷Tl>¼0?ÈJs?Rn}?IÇW?ªÇ?Ôp>Ó³Ú½áêÓ¾ç)¿åW¿­¤s¿JU¿ð}¿î·o¿¸Z¿~>¿Y¿Ó³ü¾rÂ¹¾©ßo¾¶Ïâ½4½;²Ýé=[>c>*Æ>YÕë>VÊ?yÌ?Ò#?w×.?/9?HFB?>J?8Q?­QW?p¤\?ÂHa?ýSe?Ùh?õèk?_n?mâp?är?þ¢t?b&v?zvw?ÿx?Ëy?úqz?ú/{?ªÔ{?cc|?ß|?;J}?§}?÷}?3=~?y~?Ù­~?$Û~?^?X$?ÆA?B[?Uq?s??X£?Â¯?º?ÓÃ?äË?àÒ?ìØ?)Þ?²â?æ?ê?	Ç½åV¿í0L?ñ~n><,¿e2¿á·=ç|?Û'W¿|ØS?Ô¿>ª?®W(?1ïÊ>µò?¥hg?±]?	¾·k¿b>?r²;/
¿)E?.íO¿Ú6?*ÈÖ¾ü÷½8A?è{¿<¥>ÚD?+¢M¿Î ¿¦\?¥L?lU%¿i§l¿½ç»ËBb?T]?iLÇ=ã[4¿,ä¿4¿H×½ëè?´n?Õz?ÀR>?Û¬>©í½Õ¿+O¿<w¿¿Áãj¿ÇB¿^¿¤¾Jó®½e>8!®>ôØ?Ù)?ÞF?6]?ôÌm?kx? 8~?üÿ?W~?0az?Þ:t?l?Âc?5Z?Í.P?/êE?;?íV1?ÖH'?½?Ø?¹ú
?8N?~ô>Ïlä>Õ>Ç>Ñjº>¸®>)`¢>Ût>:>
§>teu>³¨d>U>InF>½Ï8>s,>«B >_7>Gì
>ÕT>hÊð=$à=£Ð=j2Â=+À´=;¨==¹==mo|=Ýíj=t¢Z=ÜwK=.Z==ß60=§ü#=l=.=ô(=wùõ<Îæä<\Õ<É~¿ ?h¿¶õx?¦e=?E7?-\?ÕÂ¾¹
¿º?!=Ñ4u¿f§Q=_Ý@?ek?Y?öÚ>´¿û×t¿ÜË>L$+?rÿ¿W?W#¿)W?n3¿Çbh?Ç~¿}ï'?²í<>KMr¿$?~{?]¿m¼¿®K??sC?³@Ã¾\þ¿%ï¾+?óÈ~?f­5?P²î¼¤ñ5¿TW¿/U¿Üþ¸¾$¶L>é5+?q?<E~?îûZ?Lc?Ü5>À³½Ë¾»m&¿mU¿~r¿q¿;}¿k½p¿ªo[¿t1@¿5!¿EF ¿(½¾Uw¾gñ½Öd9ºÝ=¦dU>v>ûÂÃ>Åé>ß?¡û?`"?4.?4 8?yÈA?ZÐI?+ØP?uýV?öZ\?¸a?8e?¨h?Ã¾k?²mn?Âp?óÈr?üt?v?hdw?Rx?4y?1fz?Ä%{?ÐË{?·[|?kØ|?xD}?¢}?1ó}?t9~?Qv~?
«~?µØ~?C ?"?1@?äY?%p?l??¢?¯?ï¹?SÃ?uË?Ò?Ø?áÝ?sâ?iæ?×é?§×c¿²b½]ãj=ä\?FC<¸½Ò(?u9?ü¿TR?àº]¿næ$¾h¡n?mMe?©µ2?ù½G?'û}?51.?®-
¿!ÖI¿Ca?ë\¾¬¶¾c"?ú:2¿?ó¾Iñ¾ÿÔU?ûÒs¿'X>Õ£T?!b=¿¿ã÷O?-+?ß9¿âr¿Y½Z?6úc?ëÑ>0M+¿n÷¿£;¿.Þâ½J	?¹lk?2S|?CGC?$µ¹>À¹½¼þ¾&ÍK¿¢1v¿e¿¦©l¿ú>E¿£¿f«¾	%Ê½pÌ >è¨>Y{?®-'?Ç[E?¹l\?jm?ów?9~?Mþ?ÿ¨~?ù¬z?Í t?Ím?Id?ÅZ?ÄP?¤F?P-<?)ì1?£Ú'??÷?#}?Ê?÷õ>ÞLå>ÅlÖ>»aÈ>À&»>Rµ®>ä£>3>Ë>#/>ícv>ve>äU>=G>9>ÅÏ,>Ëê >Ô>R~>èÜ>ñÇñ=«á=ÿ~Ñ=9ÿÂ=à~µ=ªì¨=h8=üR=D.= z}=æk=[=ëNL=X">=*ñ0=	ª$=È<=Z=µ´=ýö<âØå<¥äÕ<Dqé¾Âr?'¿^d?Xû?£P?Ûq@?w[0¿©a)¼ªÿ=¢çÿ>«¨|¿la¹¾ù¦ã>ËL7?E ?½Y >!;¿xW¿Vy?í@ó>2z¿Ø&o?×ÎE¿Ä7?O¿ªv?¯w¿0Á?à>#5z¿?ì?,?P¿@H¿QZ>?IO?Æ¡¾(V¿<4¿&êè>"}?Ì=>?Â}<©0.¿¨l~¿*Z¿èÉ¾éâ,>]%?!n?âñ~?^?/ì?¿Y>¼½Ý5Ã¾ã/#¿R9S¿¢Jq¿û¿~¿÷}¿°ºq¿!×\¿èßA¿#¿+0¿·lÁ¾ÓÆ~¾2ÿ½*öë»kBÑ=ÂO>«ç>tÁ>z´ç>5ó?3*?È¦!?-?g8?=JA?¾aI?iwP?ï¨V?6\?oÈ`?=äd?àwh?fk?áHn?¢p?.­r?árt?¢üu?CRw?zx?{y?]Zz?{?ìÂ{?T|?¾Ñ|?°>}?}?Úî}?²5~?s~?8¨~?CÖ~?%þ~?° ?>?X?õn?d?9?Ì¡?l®?[¹?ÒÂ?Ë?Ò?EØ?Ý?5â?3æ?¨é?@R]¿/ÓB?þ*9¿Pv?6³0?Õ¶?Ôz?ñýQ>SúY¿øuf?¿®¿YÉ3?ÿf?gbh?û°q?À]{?Ìà>9O?¿L¿Ðw?ÅÌØ¾"¾ôó>¸é¿ÜÀä>Ø#ÿ½$Ë¾Þ¥f?Kh¿3­Ç=Î³b?­=+¿Í'¿"GB?ðñ9?)F¿ïw¿ Ì¾ø+R?)j?M>Ö!¿%¿ØÆB¿ÓE¾Ó÷û>Ußg?½}?}H?Æ³Æ>B¹½Â!ô¾RUH¿v³t¿2¯¿b\n¿XëG¿Þß¿4ò±¾Må½]è=Øö¢>Ô3þ>:;%?*ÕC? J[?î=l?ww?§Æ}?Ûù?FÐ~?ºöz?u?ª}m?ÙÎd?"T[?~XQ?lG?Ä<?ð2?l(??Ö?Yÿ?ÝF?+ñõ>µ,æ>W@×>B)É>â»>Ôe¯>«£>|«>þ\>1·>Vbw>+f>ÂV>´H>cR:>->æ!>Ép>[>úd>vÅò=Ïüá=ìZÒ=ÌÃ==¶=8©=¶Ý=ßì=½=Ñ~=ZÞl=p\=ú%M=ê>=u«1=jW%=$Þ=0=w@=·ø<÷Êæ<îÅÖ<t¨ ?ì&?ýÅ0¿f¾é<9?V¨O?¹ÀL>.z¿«>? ìÞ¾:ÖX?íQ¿k>6¿Ý=ËÉÖ>Æ¨>óA¾®f¿*¿DK?Ç>»êg¿oÀ|?a¿EeT?Ae¿r~?âüj¿ &Þ>.×>ÄÇ~¿2×í>ÄK>?ýUA¿¬µ&¿­ô/?¶õY?w¾f}¿é)¿uëÎ>ËÉz?ºZF?´{=e &¿³}¿±Ú^¿ýØ¾nã>¸?`ïk?&t?4a?Áa?|n>ÈVK½OÃº¾æ¿¼øP¿
p¿íc~¿ÏX~¿µ¯r¿8^¿eC¿fá$¿¿=Å¾{¾L¯¾^`¼ÊñÄ=±J>¶W>%¿>5¢å>¼?.X?ïì ?!í,?"7?Ë@?¿òH?NP?TV?0Ç[?é`?¬d?ùFh?ßik?ë#n?zp?Lr?­Zt?¢çu?@w?Éjx?Ûmy?|Nz?7{? º{?HL|?Ë|?á8}?}?~ê}?ë1~?Êo~?c¥~?ÏÓ~?ü~?Ø?=?"W?Âm?[?S?¡?À­?Æ¸?QÂ?Ê?¾Ñ?ñ×?PÝ?öá?ýå?yé?<5½¾y?~¿Uð> ?&x?ºf?eSÆ¾¬á¾Ôê?X½ñR`¿â²>¬r?¾i?ÿ?ÒÒ_?}ô">	/f¿±¶Æ¾Ðñ?é¿Ü;=Æ> -Î¾,>sAî<Ï¿K^s?0Y¿7¼<&n?5c¿^
9¿§!3?mG?z"ë¾¦Æ{¿TM¾ýõH?ro?á> ý¿~¿dI¿%à@¾ë>fd?`~?^­L?DÓ>]7#½Sdé¾ÄD¿Þs¿iá¿Òûo¿³J¿(¿¨Î¸¾Ø5 ¾Ï=7Y>siù>D#?6JB?#Z?pk?q÷v?ë}?¥ò?*õ~?t>{?gu?ôm?qSe?)â[?ìQ?®G?/Z=?C3?ý(?Ï'?v¥?\?ðÂ?Ýö>Vç>¼Ø>¦ðÉ>E¼>?°>Q¤>µF>[î>6?>¯`x>Óqg> W>ÝÛH>­;>W8.>þ:">y>a¢>	í>öÂó=ðèâ=×6Ó=ÒÄ=Eü¶=ÄOª==Â=ÅL==Öm=¢W]=	ýM=ª²?=¿e2=Ë&==±Æ=8Ì=×ù<½ç<6§×<ê¿?å$d¾Ô=Xÿa¿4ãq=0©{>ÿÏÝ¾¥l¿Êe?c·S¿+O?6²ö¾âÝo¿L	£¾=Ò» ø¾Û¼|¿à¾Ãîk?sª<ÐJ¿	»?¤$t¿Sj?½t¿Gä?ÛZ¿Ø>?õ¿ÒÖ»>ÚpN?$è0¿Bã6¿¯j ?.hc?@9¾÷Ìz¿ý¿ME´>½Áw?DÿM?RÚ=Ä¿ùj{¿'Nc¿øè¾ÃÙ=zÃ?U i?óË?øÚc?vÃ#?ûr >ÑBú¼¤A²¾¿w«N¿Ü½n¿Iü}¿°~¿ps¿l_¿à-E¿²&¿mü¿
É¾©Ï¾ßÃ¾Ö_¥¼Q¸=½wD>·Æ>ZÔ¼>»ã>©??2 ?©H,?dï6?{L@?^H?Û´O?õþU?ã|[?%G`?£sd?âh?.?k?Ðþm?>bp?Nur?aBt?Òu?À-w?ëZx?`y?Bz?à{?
±{?D|?PÄ|?3}?ÿ}?æ}?!.~?l~?¢~?XÑ~?âù~?ÿ?i;?¿U?l?Q?m?> ?­?0¸?ÐÁ?%Ê?]Ñ?×?Ý?·á?Æå?Jé?/Q?ß¯Î>£Æ¿¡¾ÅNE?Bf?'gé>_³W¿ÅtØ=ºÕ=Æ¹Í>v¯¿Ñ½ A?8Ât?¦q?.?E	¾~ç{¿Y²¾|z?ìE¿£~>àé=?p¾û>Øm:>©Û#¿sÄ{?¿F¿*¾PÙv?¿¸­H¿¤"?nÅS?KfÈ¾ud~¿Áÿ¾1??cKt?u>zÈ¿.}¿,P¿q.h¾åÅÚ>ø_?ÚH?àQ?&<à>Uk¼LÞ¾1A¿	hq¿µû¿Õq¿ÛM¿F=¿d¡¿¾¿¾fÌµ=7¶>²ô>°I!?óº@?ó÷X?3j?Dtv?H}?¬è?­?%{?PÈu?in?ÕÖe?0o\?â~R?ñCH?Èï=? ©3?Ë)?9´?Ö,?,?Ù>?ÈÈ÷>¿ëç>õæØ>ä·Ê>ÛY½>Æ°> ö¤>Þá>­>0Ç>ø^y>n_h>e~X>þªI>ïÔ;>ì.>ã">%ª>d4>u>tÀô=Õã=¿Ô=eÅ=õº·=P«=N(=£ =Ü=M=ÓÎn=¯>^=ÔN=Òz@=	 3=+²&=Ü =Ü\=úW=ö	ú<¯è<Ø<Í´?¾6j¿P!S?èAs¿Ê#¿¦{Þ¾>Úc¿ºß	¿î~?N¿ãlj?æ K½v¿V (¿T¾$¬¾Á²;¿|°}¿Np6¾\}?Ì[¾ßc#¿öw?FS~¿×Ýx?Ä}¿»¸{?^°D¿èo9>L[!?º}¿°«>å\?fò¿ø±E¿)Ö?nk?Ø'å½àðv¿Úo*¿?>	t?Ç&U?,>Ñ ¿$Uy¿tg¿«û÷¾ý =¾¬?$f?;ù?îf?À(?)fª>X;¼±©¾¦2¿¥QL¿dm¿}¿Hý~¿Ût¿.æ`¿OÍF¿(¿·Þ¿×ÓÌ¾æ¾®Ö¾Ú¼K¬=2Ð>>³4>º>zá>ý+?i²?w?®£+?/^6?öÌ??H?SO?©U?Q2[?"`?;d?äg?Sk?Ùm?áAp?4Yr?ü)t?a½u?cw?þJx?JRy?6z?üz?¨{?¸<|?½|?2-}?ì}?¹á}?R*~?6i~?¯~?ßÎ~?¾÷~?$?Í9?[T?Zk?E??u?e¬?·?MÁ?´É?ûÐ?H×?½Ü?wá?å?é?<Cm?ÿ0ü¾»
>b¹f¿ý×>aÒ>>U3¾Ö¿û?Þ¾ÃH?ýnk¿E^ï¾z*ä>ÓI?Ë|F?©)Ö>ÄØÔ¾6Ú~¿sã= f?d¿Ô»á>¥Ò½E°p½ðG½þT©>D>¿)²?ð>1¿Õ!|¾Y³|?y¶Ö¾V¿Rï?©^?À¤¾AÅ¿Ã¦¾â_4?5x?ñ³>G=¿Ì{¿>.V¿¾ºÉ> [?ûÂ?^U?ýÀì>òû5<<Ó¾ËY=¿$o¿þ¿K s¿¢O¿^¿jÆ¾ÓA¾x=	>¶¾ï>³J?j'??ªÈW?úÉi?íu?ù}?ðÛ?Ì7?ÍÇ{?_'v?Ün?Yf?7û\?ÕS?­ØH?Ñ>?<4?*?U@ ?÷³?Ç?º?0´ø>ñÊè>ºÙ>ÿ~Ë>T¾>Ív±>¥>ø|>ò> O>0]z>üLi>@\Y>zJ>*<>Ô />#>ÎF>dÆ> ý>í½õ=)Áä=¥îÔ=b2Æ=£y¸=Ú²«=Í=º=Dk=pÒ=Ço=¼%_=$«O=úBA=RÚ3=_'=7Â=ó=»ã=û<2¡é<ÇiÙ<uFÀ¾Ê^¿<w?ÕÝ¾-Ö}¿àci¿*|¿¸=xðH?f¿R
?NÉ>ÐLb¿¾,e¿¬|¿½¬!¿Wh¿Ôh¿Û³Á= l~?Là¾è¾±Æe??¿Á?·Û¿q?§E+¿G=;·8?\x¿¸ß#>Ýfh?¿eS¿J¥ü>´kr?Êc-½K
r¿o©5¿¥²z>I«o?éÌ[?fÝH>>9¿Ýv¿ãJk¿¿¶0=Gu?üb?õû?³!i?I,?ñF´>ñ(û;³¡¾qÈ¿lëI¿ãþk¿Y
}¿r@¿í\u¿I3b¿¨gH¿XH*¿e¾	¿(Ð¾$4¾¢ç¾~Ý½Põ='9>­¡>/¸>4dß>¹=?©Þ?¼?1þ*?Ì5?M??s£G?ëðN?¿SU?yçZ?âÄ_?.d?$³g?Méj?*´m?d!p?ý<r?~t? ¨u?ów? ;x?lDy?*z?òz?{?ä4|?Æ¶|?R'}?Õ}?PÝ}?&~?æe~?Ñ~?cÌ~?õ~?G?08?õR?$j?9~??¬?·«?·?ËÀ?CÉ?Ð?ôÖ?tÜ?8á?Xå?êè?©/=>ÿ}¿Zg?4Ëo¿é³¿!»¾÷<¿ß[M¿wSq?>S¿Cµ{?¨'¿pÓI¿
'=G?j??_¬ë=2a)¿ôÎn¿ 1·>.WE?üw¿Y?òs¾æ3ó=?1J¾ qñ>ùÆT¿?¿\g¶¾£?¸<§¾ôb¿%Hü>Âh?æ¾Uç¿QÓÅ¾ )?óV{?vÄÌ>ÖÄð¾z~x¿Óæ[¿Ô¾]i¸>
W?¤û?âoY?aù>~×=¹hÈ¾É9¿a´m¿è¿et¿ÜR¿;u ¿I(Í¾½(¾=Þ`>£Þê>¢G?¥=?<V?Þðh?Gcu?Å¼|?pÌ?U?k	|?µv?®No?üÙf?;]?ó¡S?ºlI?I??yÏ4?®*?"Ì ?×:?.?-6?Sù>ë©é>ßÚ>õEÌ>°Ð¾>ð&²>hA¦>>*¢>×>X[{>|:j>:Z>$IK>^W=>
U0>(3$>sã>aX>'>b»ö=@­å=ÊÕ=(ÿÆ=P8¹=bd¬=âr =dT=ú=ÇW=I¿p=È`=1P=!B=4=ë(=c=2=|o=4ü<Fê<KÚ<Ï{¿Sºÿ½*Ú>F³>bP¿\lv¿¸¡-¿Ú?Úª>¿§¸:>vA?É|¿ña¿X-[¿¹Z¿¢L~¿Zô?¿<v¸>o?#¿3~¾ôîI?÷àv¿E0~?¾õz¿¨¿a?X¿e­½é;M?4o¿UZ=bõq?Õ"î¾øÃ^¿}ú×>(çw?XÀà<l¿=9@¿æB>¢j?¤ía?%v>×¿t¿pÑn¿Zê
¿ýó<<t?©_?"Ô?êk?ôk0?C¾>FOÛ<Óh¾½S¿ñxG¿Àj¿|¿y¿0v¿´yc¿ßüI¿Ô,¿n¿]Ô¾Wã¾¤ö"¾:r"½=|3>«>iÛµ>,MÝ>ßN ?W
? ?3X*?]:5?¦Ì>?ë2G?oN?¯ýT?ZZ?d_?"Éc?}g?¾j? m?Ç p?« r?èøs?Êu?pöv?ò*x?6y?~z?çz?ñ{?-|?÷¯|?k!}?¸}?âØ}?ª"~?b~?ð~?åÉ~?nó~?h?6?Q?íh?+}?³?â?«?k¶?GÀ?ÒÈ?7Ð?Ö?*Ü?øà? å?ºè?^':¿ß]1¿?i?ÀÈÊ¾8×z¿Ï}S¿EÒ~¿ÜH¤¾ÿµy?¦I¿ßJt?l¾-x¿~¢¾#5>°S>VL¾ÆY¿Ì÷L¿÷å?7?!â¿b8C?ÎÄâ¾Å>&e¶¾6ë?³ìf¿rñy?Báü¾ÎOì¾Æ?c.l¾[µl¿¤ÍÔ>Kp?¦~4¾Ê~¿ ä¾(?à¬}?då>ü|Ú¾su¿P;a¿|Û­¾¸Ù¦>0R?Æò?Q]?ú?~=\.½¾¨5¿õ³k¿»¿¶u¿\sT¿±#¿»ÛÓ¾16¾Ý~S=é®>÷å>@?¬ó;?±]U?ãh?zÕt?jr|?.º?äp?þH|?Pàv?N¿o?¾Yg?>^?<2T? J?2­??ôa5?=+?W!?vÁ?a?±?2ú>­ê>_Û>ÆÍ>î¿>ûÖ²>°æ¦>ý²>U3>â^>pY|>ð'k>Ö[>+L>>>:	1>-Û$>>\ê>->Ô¸÷=Uæ=h¦Ö=ëËÇ=ûö¹=ê­=*¡=Cî=¿=Ý=·q=Ôó`==YQ=HÓB=äN5=Kº(=í=]==û=Sý<Yë<W,Û<%¼/¿8?sÓ¾°k?ÇL¾b?¿ùKÄ½´vr?a¾Â«½¾Vw?µSt¾óÃr¿ö{¿¥z¿ïÙz¿R¿5`?M>P?UÞL¿ðM÷¼é%?ne¿oÍt?w4o¿~L?ÿÜ¾Êm]¾^?
c¿>â]½y?	ýÂ¾*×h¿Öè±>åý{?ÌÆ=ì3e¿gJ¿½¯	>aõd?=g?ºu>]ó¾Êp¿xr¿$¿g¤¼­© ?*\?È?7âm?Òx4?ÍÇ>èß;=±¾ÆÔ¿[úD¿Ji¿Uê{¿¨¿éûv¿f¹d¿êK¿Ï-¿Éu¿SØ¾p¾*¾4=½SE=zÐ->±x>3³>ú4Û>ß¾þ>t5?C?´±)?Â§4?ÜK>? ÂF?+N?O§T?öPZ?¨A_?ßc?¦Og?Äj?ñhm?	ào?<r?9às?^}u?Úãv?Ôx?(y?_z?Ýz?Ö{?$%|? ©|?~}?~}?pÔ}?Ï~?=_~?~?dÇ~?Cñ~??ñ4?%P?µg?|?É??Xª?Óµ?Ä¿?_È?ÔÏ?IÖ?àÛ?·à?éä?è?tx¿Y(>Øè>ØºÆ>yhZ¿u½~¿»ÊY¿cV>}05?-©f¿ð4?´¨C>Ô{¿ïè'¿¸>¾Jü¾Íý¾üqw¿WÚ¿|D?^+Ê>­^{¿.ã`?ê¿èê>øZ¿Py7?Vt¿,]p?ÛÔÃ¾6¿©|?8¾d¾t¿ßº«>ümv?]ñÐ½3p|¿ê¨ ¿Xª?~5?Ë«ü>°Ã¾çq¿H)f¿àÀ¾>ØM?d¨?#a?°?:³=ÄØ±¾å1¿i¿¢u¿2óv¿÷ÈV¿:&¿
Ú¾BC¾6¸ =[ø>Ï	á>x5?S:?"T?3g?(Dt?é$|?)¥?Ú?|?1:w?{.p?HØg?=^?¯ÁT?ÂJ?@@?øó5?¼Ì+?Îâ!?ÖG?_?×,?Ëtû>7gë>2Ü>sÓÍ>GÀ>î³>ã§>éM>tÄ>´æ>wW}>Vl>õ[>)çL>²Ù>>e½1>.%>²>S|>0>B¶ø=gç=G×=¬È=¥µº=pÇ­=r½¡=!=ü=rb=»¯r=ßÚa=H0R=oC=,	6=ªg)=G¦=µ=ý=qþ<mwì<Ü<£Åv>B|?2v¿ãík?,?AðÊ=Ó?2v?_Ù4¿aÞ>ºó5¿H{?8>Ñ=A¿D{¿Ä}¿Ç^¿Ë@¾?K?éû#?2k¿ÜßA>¸¢ô>¾ÎK¿	²c?ê\¿ó2?Æ¾ë1°¾Íl?ùS¿aÑ$¾x¿}?J'¾¥+q¿¯>	ª~?%8*>ûQ]¿´4S¿Jà =º§^?Ql?p§>¬(à¾1m¿ãèt¿ª2¿W¸S½Ã0ô>ÊX?õ?Ep?2o8?MpÑ>N=Âî¾ÌK¿ÏoB¿g¿I{¿Í¿Ä¾w¿Tòe¿¿M¿Q/¿mM¿ØÛ¾a;¾y1¾$W½ÁÖu=#(>Äâ>ç/±>¡Ù>ÖÞü> `?d?´
)?±4?¥Ê=?´PF?oÈM?¢PT?MZ?®ÿ^?fVc?g?@gj?Cm?+¿o?±çq?rÇs?Ýgu?2Ñv?¥
x?}y?5z?Òz?²{?8|?C¢|?}?ny}?ùÏ}?ñ~?ã[~?%~?àÄ~?ï~?¥?O3?»N?{f?{?Þ?K?¨©?:µ??¿?íÇ?qÏ?óÕ?Û?wà?±ä?Zè?ó§¤¾¦c?qÒ¿go?Ó¾>ä2¿¤4¸¾T E?uc>7L¿+>Ïý?°íR¿üe¿v[¿Íè¾}=¿Î×¿(?¾¾g?,>@ªj¿¾u?ôÍ>¿£p?ú#¿Ê¯P?¾|¿Ib?pÅ¾'¿Év?½@©z¿¡^>:6{?ãÞ¼PÛx¿,À¿a¥?ï?Ï	?Hm¬¾Üm¿|®j¿XÓ¾¤>ÇG??Úd?.{?#ç=i¦¾ÿl-¿gg¿\¿Qx¿Y¿§)¿Ý á¾ Q¾qÖÛ<Ëzv>XÜ>{&?C¯8?UâR?]Nf?Q¯s?CÔ{?b?n ?Â|?Uw?2p?Uh?9!_?KPU?¾$K?NÓ@?6?[,?­m"?ôÍ?)?ì§?_ü>Eì>kÝ>ûÎ>Á>É6´>1¨>Äè>U>{n>mU~>®m>@Ó\>¶M>Ñ?>q2>*+&>K¹>G>1>¬³ù=uqè="^Ø=leÉ=Mt»=õx®=¸b¢=ÿ!=8¨=Çç=ó§s=éÁb=SS=cD=tÃ6=	*=¢G=²K=¾	=ÿ<ií<çîÜ<fr?y0ê>& U¿¸e·>v?U7?Ûn?ï¹"?¾y¿uS?õu¿L?Ð?é­ä¾èY¿d¿B",¿n=Í«m?JDÚ>Â\|¿,¡Ì>ðÇ>¬*¿1hK?D¿½F?¬Ö"¾î¾ÆÕv?YB¿¶!¾àÜ?Ó	P¾o±w¿ËE>½ç?%6p>4T¿[¿{·<¾W?Ðq?K½>ÀÌ¾;i¿´ww¿ï ¿$yª½
Øæ>|¯T?½]~?År?N<?íüÚ>9
¬=úA|¾¹¿uÙ?¿°ìe¿rz¿fè¿+yx¿w$g¿UN¿?G1¿Q"¿ß¾ä¾ 8¾À$r½ ]=8t">éK>Ø®>$×>§ýú>ý?ÀÈ?5c(?)3?I=?ßE?ëdM?¦ùS?]¹Y?w½^?¶c?hëf?;j?$m?-o?	Ëq?®s?GRu?x¾v?fúw?gy?þùy?øÇz?z{?C|?^|?}?Bt}?}Ë}?~?X~?;~?ZÂ~?æì~?À?«1?OM?@e?üy?ò??÷¨?¡´?º¾?zÇ?Ï?Õ?KÛ?6à?yä?)è?º}?ûåe?¼:~¿«g?Alë>rA½é(>ß?2ò©¾Á½E¾©ûc?m¿Í\¿ÒPK¿R¾9¿«i¿¼>r¿ºpÚ½Ð{?Õâ½BN¿?¿[¿Hø<?OB¿åõd?0÷¿¼¥P?ï>¾j?<¿Àn?c¤=Ôh~¿`,>]~?øC=Qt¿a=¿ÅJì>~Ú? ç?dÁ¾bTi¿ßÈn¿7å¾ÕÚa>79B?pO~?ëÌg?Q@?Òd>kâ¾|:)¿óe¿=£~¿[1y¿ØF[¿Ên,¿Ù±ç¾_Y^¾gl<wüj>_×>?å7?Q?Úee?øs?z{?Ør?´?vû|?¾èw?sq?µÑh?2¨_?ÞU?¶K?eA?7?íé,?<ø"?ÑS?½	?Ö"?,Iý>¡#í>ÖÝ>]`Ï>õ¼Á>æ´>Ö¨>>æ>8ö>SS>ùïm>å°]>
N>é[@>©%3>!Ó&>âU>8 >/¥>±ú=]é=û9Ù=)2Ê=ó2¼=x*¯=ý£=Û»=s7=m=* t=ò¨c=^ÞS=º+E=¼}7=hÂ*=üè=Üá=~	=W =[î<.ÐÝ<?H?B<á¾ßsð½ÚÙ¾Uc?jC?*ïu?õ¬=w|q¿;E?\6{¿{äè>²Z?q½5¿)+0¿mWÑ¾Ø¥>'~?±ï<>òf¿á2?Ý²=±S¿´,?
ä&¿Så>U¼ãT¿ëV}?t}-¿®H¼¾Òk?jðã½\|¿Î1ç=7µ?h>ËJ¿Zc¿:
½=P?õt?=¦Ò>¸¸¾éd¿²y¿äÅ&¿yêê½%LÙ>Z´P?=}?kt?{@?éqä>Ó=j¾Í¿v7=¿·Id¿`äy¿-ù¿+y¿ÄOh¿ P¿>ý2¿lô¿yDã¾ ¾{?¾`X½LhD=Ä>Hh>¬>åÔ>Tù>j³?
?6»'?,í2?öÆ<?ølE?M?\¢S?)mY?{^?Ðâb?¹f?ºj?÷l?}o?F®q?s?<u?ª«v?êw?Bþx?»íy?W½z?Mq{?G|?s|?	}?o}?ýÆ}?)~?%U~?N~?Ñ¿~?´ê~?Ù?0?âK?d?ëx??±?E¨?´?5¾?Ç?©Î?FÕ? Û?õß?Aä?ùç?Å¬~?<>;¿4´£>Þq? &?½SM?Ä1^?
¤H¿0®Ý>»!¿ô?3ÿ½7Ûr¿%vu¿×i¿&~¿ÒO¿A1*>âb?Óý¾hy(¿9~?±p¿)X?@öZ¿ÊÐs?ñ}¿¿;?z¥¼ØûN¿Yb?¼y*>Òô¿µB¨=LÝ?Ìú=/n¿ú)¿ómÐ>[ö~?Ô?kvy¾ Qd¿vr¿ ÷¾J==>öq<?5A}?åj?ÿè?0'>ùD¾àò$¿(¶b¿P~¿82z¿În]¿uS/¿¨6î¾¨k¾óc;v_>
Ò>òü?yZ5?ÈVP?yd? {r?){?U?hÆ?Û2}?i=x?=sq?Li?%.`?üjV?FL?#÷A?8§7?ów-?z#?mÙ???ô2þ>î>¨Þ>&Ð>»wÂ>7µ>	{©>K>w>ë}>(>7Ýn>^>îSO>úA>ÂÙ3>{'>tò>'2>+->t®û=Iê=ÒÚ=äþÊ=ñ¼=úÛ¯=A­£=·U=®Æ=oò=`u=ûd=hµT=ßóE=88=Æo+=V=x=>*
=e =¥Mï<v±Þ<ãÐ½¥{¿-.?Ör¿7@¬>P?	å?ØIþ¾þ¿¶Âf?^óF¿-<3~?Åò¡>%d¾]Ð¾ÐÖ½è|?~p|?Iº½%t¿ê¿@?»èð½²e®¾l?6¢¿>>Á.¿¡ò?ô¤¿Cî¾om|?óP¼q"¿ÆG=¿~?1-¼>$:@¿ÕÓi¿6¸½A*H?Ix?¬ç>¬I¤¾d=`¿+{¿F-¿Ö¾Ë>(L?|?ñÅu?rÆC?<Îí>+äù=ÏX¾Hw ¿û:¿¾b¿í y¿Þÿ¿Ôy¿3ti¿Q¿E¯4¿´Ã¿õæ¾Ä.¤¾s!F¾ì½3®+=³>ó6z>&ª>ÉÈÒ>Þ7÷>JÜ?àK?¸'?ºX2?}D<?úD?ÞL?ÄJS?¯ Y?P8^?´¨b?kf?¸ãi?ÄÐl?Ð[o?fq?|s?Ù&u?Êv?¸Ùw?ðx?káy?«²z?h{?C|?|?}?Ùi}?xÂ}??~?ÁQ~?^~?F½~?è~?ñ?^.?tJ?Çb?Øw??ã?§?l³?¯½?Æ?EÎ?ïÔ?´Ú?´ß?ä?Èç?flç>w¸-¿Êì1=ì¾gk?"úw?'ý?¾4Ý>Ë~¿ÓSS?¿yk¿¬Ah?¹¸>òmA¿<¿2¿?Oz¿i¿ vÚ>L¬r?;2¿eô¾½r?Fß|¿¬Gm?±Ñm¿xæ|? ¶v¿b-"?´cÔ=tö^¿ºT?6>ÍI¿cb»e´?/I>_ñf¿05¿WÎ³>ÞC}?.Ø)?¥ÒH¾ÜÖ^¿çµu¿4¿'_>s6?'ò{?7Êm?"t?à@>ç¾´ ¿ã8`¿£q}¿Ô{¿?_¿|-2¿ñ®ô¾íx¾í6*¼!èS>Í>â?ª3?ÿ
O?fc?ÉÛq?Ïz?5?ÐÕ?4h}?Wx?Üq?<Æi?³`?÷V?ÖL?2B?\78?.?i$?È^?G
?(?uÿ>)ßî>Zzß>²ìÐ>c2Ã>ÉE¶>îª>ö¸>p>>v§>gÊo>l_>È"P>ÞA>Ö4>#(>>Ä>$µ>Ó«ü=5ë=¦ñÚ=ËË=;°½={°=R¤=ï=çU=Ãw=v=we=qU=¼F=Jò8=$,=¯+ =0=þµ
=t=·?ð<½ß<S[d¿÷<¿$Â?2c¿jÊ¾Y~>Ì²¼×ßf¿ÂHÜ½Ý|?ãÞÈ¾7W×¾RÏt?{±'?c
=Õ?¢½]´V>ÉVK?ãg?%£¾î9[¿Ëó`?Ý£¢¾>¾0À>a½¾aÞ>Ãª>¼F¿¢~?û¾d¿¿_év?¾=bÿ¿±D½°{?|ëÜ>½Ù4¿Ê±o¿µG¾??V{?cü>Ù²¾Ö8[¿g&}¿ü3¿'5¾å¦½>±FH?ôj{?gw? ^G?ã÷>öW>úF¾ù¾,Ñ7¿Ûß`¿$Rx¿wü¿buz¿¼j¿1S¿K]6¿!¿É¡ê¾Ð§¾ô"M¾àß ½ò=`>Ýu>Ì§>ðªÐ>HSõ>?¥?¼i&?ÓÃ1?Á;?·D?U8L?ÞòR?ïÓX?`õ]?anb?¥Sf?·i?]ªl?q:o?jtq?^cs?u?×v?HÉw?Îáx?Õy?ô§z?Ä^{?7ý{?|?ý|?d}?ï½}?Q~?ZN~?l~?¸º~?Jæ~??µ,?I?a?Äv?'??ß¦?Ñ²?)½?Æ?àÍ?Ô?iÚ?sß?Ðã?ç?â¢¿Â~¿rJ?x´u¿[Î>7Þf?qJ?j§ ¾À f¿Ã@?ù¿þ¬!?j(?D1å¾Zg¿©\w¿¤4]¿i»³¾ÔÅ'?BIV?\ÿ3¿þý¾ñ]?¾Ø¿[z?½-z¿åÿ?lj¿Òf?Ûpg>Âøk¿~øC?Ö·>?i|¿ïý¹½¢â}?Ôë>Æ­^¿»@¿±>hÄz?~¡3?ü´¾÷æX¿ex¿¶³¿ æ=>0?bz?zp?©à$?lpZ>Ço¾&¿j£]¿Eµ|¿÷{¿a¿±ü4¿\û¾¾Àº¼êRH>äÈ>\Ä?õ1?<»M?~b?ø8q?Srz?°?Óâ?}?áx?kDr?§>j?ü6a?LW?µeM?­C?	Ç8?Ö.?$?áã?<??Ø ?¼ï>øKà>¥²Ñ>íìÃ>Bõ¶>¿Äª>S>N>0>O&>·p>I`>ñP>B>äA5>ëÊ(>+>úU>=>-©ý=!ì=wÍÛ=TÌ=Ün¾=û>±=Æ÷¤=l=!å=ý=Éw=
^f=ycV=(G=¬9=Ê,=	Í =Y¤=¾A==É1ñ<tà<î¿\¿ë×=<?£¼¾\j¿;Ý¾ú±¿lÔ|¿ÔÎà>¹W=¯p«=|F¿é¸@?'ëd?u3Û>7á>VÛ ?Wµo?L\A?ò¿	6¿±>v?ø¿îÀ=	mR>¼!Y¾Âé:»Í>áY¿(`y?2Æ¾g·$¿½ín?ÔÜ*>/ñ~¿ÿX¾qv?{ü>,¶(¿½¯t¿ûM¾´e6?Ê,}?Òù?:µu¾ÝÝU¿;_~¿V¬9¿JU¾«¯>ÉÕC?z?äx?´ÜJ?o ?W°#>ó5¾ú"ò¾55¿&_¿xw¿øî¿·{¿U¨k¿c~T¿F8¿ªY¿Jî¾öo«¾æ!T¾!®½Øjô<J¬>Ïo>p¥>ÿÎ>mó>b,
?äÌ?BÀ%?w.1?K>;?D?uÓK?«R?êX?3²]?Ù3b?¯ f?6i?Ñl?òo?RWq?Js?ût?Òrv?È¸w?Óx?§Èy?3z?rU{?"õ{?|?s÷|?[_}?a¹}?_~?ïJ~?v~?(¸~?ä~?
?+?G?H`?¯u?7?D?+¦?5²?¢¼?©Å?{Í?@Ô?Ú?1ß?ã?eç?

NoOpNoOp

Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*¸
value­B© B¡
±
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
°
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
À

0beta_1

1beta_2
	2decay
3learning_rate
4itermmmmmmmmmmmmmmmm  m¡!m¢"m£v¤v¥v¦v§v¨v©vªv«v¬v­v®v¯v°v±v²v³ v´!vµ"v¶*
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
¦
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
wq
VARIABLE_VALUE7transformer_model_3/positional_embedding/dense_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5transformer_model_3/positional_embedding/dense_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEItransformer_model_3/transformer_encoder/multi_head_attention/query_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEGtransformer_model_3/transformer_encoder/multi_head_attention/key_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEItransformer_model_3/transformer_encoder/multi_head_attention/value_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENtransformer_model_3/transformer_encoder/multi_head_attention/projection_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUELtransformer_model_3/transformer_encoder/multi_head_attention/projection_bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAtransformer_model_3/transformer_encoder/layer_normalization/gamma&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE@transformer_model_3/transformer_encoder/layer_normalization/beta&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5transformer_model_3/transformer_encoder/conv1d/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3transformer_model_3/transformer_encoder/conv1d/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_model_3/transformer_encoder/conv1d_1/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_model_3/transformer_encoder/conv1d_1/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUECtransformer_model_3/transformer_encoder/layer_normalization_1/gamma'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEBtransformer_model_3/transformer_encoder/layer_normalization_1/beta'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE transformer_model_3/dense/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtransformer_model_3/dense/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"transformer_model_3/dense_1/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE transformer_model_3/dense_1/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
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
¦
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
¦
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

kernel
 bias*
¥
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
 trace_1* 
* 

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§dropout
query_kernel

key_kernel
value_kernel
projection_kernel
projection_bias*
¬
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses
®_random_generator* 
¶
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses
	µaxis
	gamma
beta*
Ï
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
º__call__
+»&call_and_return_all_conditional_losses

kernel
bias
!¼_jit_compiled_convolution_op*
¬
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses
Ã_random_generator* 
Ï
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses

kernel
bias
!Ê_jit_compiled_convolution_op*
¶
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses
	Ñaxis
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
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

×trace_0* 

Øtrace_0* 
* 
* 
* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

Þtrace_0
ßtrace_1* 

àtrace_0
átrace_1* 
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
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses*
* 
* 
¬
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses
í_random_generator* 
* 
* 
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
¨	variables
©trainable_variables
ªregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses* 
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
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses*
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
ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
¶	variables
·trainable_variables
¸regularization_losses
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses* 
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
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses*
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
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses*
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


§0* 
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
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses* 
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

VARIABLE_VALUE>Adam/transformer_model_3/positional_embedding/dense_2/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/transformer_model_3/positional_embedding/dense_2/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
«¤
VARIABLE_VALUENAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
²«
VARIABLE_VALUEUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¥
VARIABLE_VALUEHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/transformer_model_3/transformer_encoder/conv1d/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
¨¡
VARIABLE_VALUEJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
§ 
VARIABLE_VALUEIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/transformer_model_3/dense/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/transformer_model_3/dense/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/transformer_model_3/dense_1/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/transformer_model_3/dense_1/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE>Adam/transformer_model_3/positional_embedding/dense_2/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/transformer_model_3/positional_embedding/dense_2/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
«¤
VARIABLE_VALUENAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
­¦
VARIABLE_VALUEPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
²«
VARIABLE_VALUEUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
°©
VARIABLE_VALUESAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¥
VARIABLE_VALUEHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUEGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE:Adam/transformer_model_3/transformer_encoder/conv1d/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
¨¡
VARIABLE_VALUEJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
§ 
VARIABLE_VALUEIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/transformer_model_3/dense/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/transformer_model_3/dense/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/transformer_model_3/dense_1/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/transformer_model_3/dense_1/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
Â
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17transformer_model_3/positional_embedding/dense_2/kernel5transformer_model_3/positional_embedding/dense_2/biasConstItransformer_model_3/transformer_encoder/multi_head_attention/query_kernelGtransformer_model_3/transformer_encoder/multi_head_attention/key_kernelItransformer_model_3/transformer_encoder/multi_head_attention/value_kernelNtransformer_model_3/transformer_encoder/multi_head_attention/projection_kernelLtransformer_model_3/transformer_encoder/multi_head_attention/projection_biasAtransformer_model_3/transformer_encoder/layer_normalization/gamma@transformer_model_3/transformer_encoder/layer_normalization/beta5transformer_model_3/transformer_encoder/conv1d/kernel3transformer_model_3/transformer_encoder/conv1d/bias7transformer_model_3/transformer_encoder/conv1d_1/kernel5transformer_model_3/transformer_encoder/conv1d_1/biasCtransformer_model_3/transformer_encoder/layer_normalization_1/gammaBtransformer_model_3/transformer_encoder/layer_normalization_1/beta transformer_model_3/dense/kerneltransformer_model_3/dense/bias"transformer_model_3/dense_1/kernel transformer_model_3/dense_1/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_344676
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¾*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameKtransformer_model_3/positional_embedding/dense_2/kernel/Read/ReadVariableOpItransformer_model_3/positional_embedding/dense_2/bias/Read/ReadVariableOp]transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/Read/ReadVariableOp[transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/Read/ReadVariableOp]transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/Read/ReadVariableOpbtransformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/Read/ReadVariableOp`transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/Read/ReadVariableOpUtransformer_model_3/transformer_encoder/layer_normalization/gamma/Read/ReadVariableOpTtransformer_model_3/transformer_encoder/layer_normalization/beta/Read/ReadVariableOpItransformer_model_3/transformer_encoder/conv1d/kernel/Read/ReadVariableOpGtransformer_model_3/transformer_encoder/conv1d/bias/Read/ReadVariableOpKtransformer_model_3/transformer_encoder/conv1d_1/kernel/Read/ReadVariableOpItransformer_model_3/transformer_encoder/conv1d_1/bias/Read/ReadVariableOpWtransformer_model_3/transformer_encoder/layer_normalization_1/gamma/Read/ReadVariableOpVtransformer_model_3/transformer_encoder/layer_normalization_1/beta/Read/ReadVariableOp4transformer_model_3/dense/kernel/Read/ReadVariableOp2transformer_model_3/dense/bias/Read/ReadVariableOp6transformer_model_3/dense_1/kernel/Read/ReadVariableOp4transformer_model_3/dense_1/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpRAdam/transformer_model_3/positional_embedding/dense_2/kernel/m/Read/ReadVariableOpPAdam/transformer_model_3/positional_embedding/dense_2/bias/m/Read/ReadVariableOpdAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/m/Read/ReadVariableOpbAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/m/Read/ReadVariableOpdAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/m/Read/ReadVariableOpiAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/m/Read/ReadVariableOpgAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/m/Read/ReadVariableOp\Adam/transformer_model_3/transformer_encoder/layer_normalization/gamma/m/Read/ReadVariableOp[Adam/transformer_model_3/transformer_encoder/layer_normalization/beta/m/Read/ReadVariableOpPAdam/transformer_model_3/transformer_encoder/conv1d/kernel/m/Read/ReadVariableOpNAdam/transformer_model_3/transformer_encoder/conv1d/bias/m/Read/ReadVariableOpRAdam/transformer_model_3/transformer_encoder/conv1d_1/kernel/m/Read/ReadVariableOpPAdam/transformer_model_3/transformer_encoder/conv1d_1/bias/m/Read/ReadVariableOp^Adam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/m/Read/ReadVariableOp]Adam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/m/Read/ReadVariableOp;Adam/transformer_model_3/dense/kernel/m/Read/ReadVariableOp9Adam/transformer_model_3/dense/bias/m/Read/ReadVariableOp=Adam/transformer_model_3/dense_1/kernel/m/Read/ReadVariableOp;Adam/transformer_model_3/dense_1/bias/m/Read/ReadVariableOpRAdam/transformer_model_3/positional_embedding/dense_2/kernel/v/Read/ReadVariableOpPAdam/transformer_model_3/positional_embedding/dense_2/bias/v/Read/ReadVariableOpdAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/v/Read/ReadVariableOpbAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/v/Read/ReadVariableOpdAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/v/Read/ReadVariableOpiAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/v/Read/ReadVariableOpgAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/v/Read/ReadVariableOp\Adam/transformer_model_3/transformer_encoder/layer_normalization/gamma/v/Read/ReadVariableOp[Adam/transformer_model_3/transformer_encoder/layer_normalization/beta/v/Read/ReadVariableOpPAdam/transformer_model_3/transformer_encoder/conv1d/kernel/v/Read/ReadVariableOpNAdam/transformer_model_3/transformer_encoder/conv1d/bias/v/Read/ReadVariableOpRAdam/transformer_model_3/transformer_encoder/conv1d_1/kernel/v/Read/ReadVariableOpPAdam/transformer_model_3/transformer_encoder/conv1d_1/bias/v/Read/ReadVariableOp^Adam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/v/Read/ReadVariableOp]Adam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/v/Read/ReadVariableOp;Adam/transformer_model_3/dense/kernel/v/Read/ReadVariableOp9Adam/transformer_model_3/dense/bias/v/Read/ReadVariableOp=Adam/transformer_model_3/dense_1/kernel/v/Read/ReadVariableOp;Adam/transformer_model_3/dense_1/bias/v/Read/ReadVariableOpConst_1*Q
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
__inference__traced_save_345738
ç
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename7transformer_model_3/positional_embedding/dense_2/kernel5transformer_model_3/positional_embedding/dense_2/biasItransformer_model_3/transformer_encoder/multi_head_attention/query_kernelGtransformer_model_3/transformer_encoder/multi_head_attention/key_kernelItransformer_model_3/transformer_encoder/multi_head_attention/value_kernelNtransformer_model_3/transformer_encoder/multi_head_attention/projection_kernelLtransformer_model_3/transformer_encoder/multi_head_attention/projection_biasAtransformer_model_3/transformer_encoder/layer_normalization/gamma@transformer_model_3/transformer_encoder/layer_normalization/beta5transformer_model_3/transformer_encoder/conv1d/kernel3transformer_model_3/transformer_encoder/conv1d/bias7transformer_model_3/transformer_encoder/conv1d_1/kernel5transformer_model_3/transformer_encoder/conv1d_1/biasCtransformer_model_3/transformer_encoder/layer_normalization_1/gammaBtransformer_model_3/transformer_encoder/layer_normalization_1/beta transformer_model_3/dense/kerneltransformer_model_3/dense/bias"transformer_model_3/dense_1/kernel transformer_model_3/dense_1/biasbeta_1beta_2decaylearning_rate	Adam/itertotal_2count_2total_1count_1totalcount>Adam/transformer_model_3/positional_embedding/dense_2/kernel/m<Adam/transformer_model_3/positional_embedding/dense_2/bias/mPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/mNAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/mPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/mUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/mSAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/mHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/mGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/m<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/m:Adam/transformer_model_3/transformer_encoder/conv1d/bias/m>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/m<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/mJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/mIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/m'Adam/transformer_model_3/dense/kernel/m%Adam/transformer_model_3/dense/bias/m)Adam/transformer_model_3/dense_1/kernel/m'Adam/transformer_model_3/dense_1/bias/m>Adam/transformer_model_3/positional_embedding/dense_2/kernel/v<Adam/transformer_model_3/positional_embedding/dense_2/bias/vPAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/vNAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/vPAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/vUAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/vSAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/vHAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/vGAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/v<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/v:Adam/transformer_model_3/transformer_encoder/conv1d/bias/v>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/v<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/vJAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/vIAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/v'Adam/transformer_model_3/dense/kernel/v%Adam/transformer_model_3/dense/bias/v)Adam/transformer_model_3/dense_1/kernel/v'Adam/transformer_model_3/dense_1/bias/v*P
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
"__inference__traced_restore_345952©
£

õ
C__inference_dense_1_layer_call_and_return_conditional_losses_344060

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²|

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_343996

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢'multi_head_attention/add/ReadVariableOp¢1multi_head_attention/einsum/Einsum/ReadVariableOp¢3multi_head_attention/einsum_1/Einsum/ReadVariableOp¢3multi_head_attention/einsum_2/Einsum/ReadVariableOp¢3multi_head_attention/einsum_5/Einsum/ReadVariableOp²
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ô
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NI,HIO->...NHO¶
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHO¶
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
: ®
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'multi_head_attention/dropout_1/IdentityIdentity&multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/Identity:output:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...HNM,...MHI->...NHI¶
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0û
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0¸
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_2/IdentityIdentitymulti_head_attention/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:É
 layer_normalization/moments/meanMeandropout_2/Identity:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/Identity:output:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ç
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75½
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
#layer_normalization/batchnorm/mul_1Muldropout_2/Identity:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ·
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:Ã
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout_3/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:É
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ë
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
©
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_345106
xQ
>positional_embedding_dense_2_tensordot_readvariableop_resource:	K
<positional_embedding_dense_2_biasadd_readvariableop_resource:	
positional_embedding_344960f
Ntransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource:S
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
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢3positional_embedding/dense_2/BiasAdd/ReadVariableOp¢5positional_embedding/dense_2/Tensordot/ReadVariableOp¢1transformer_encoder/conv1d/BiasAdd/ReadVariableOp¢=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp¢?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢@transformer_encoder/layer_normalization/batchnorm/ReadVariableOp¢Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp¢Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp¢Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp¢;transformer_encoder/multi_head_attention/add/ReadVariableOp¢Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp¢Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp¢Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp¢Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpK
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
valueB:º
"positional_embedding/strided_sliceStridedSlice#positional_embedding/Shape:output:01positional_embedding/strided_slice/stack:output:03positional_embedding/strided_slice/stack_1:output:03positional_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskµ
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
value	B : ¯
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
value	B : ³
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
valueB: Å
+positional_embedding/dense_2/Tensordot/ProdProd8positional_embedding/dense_2/Tensordot/GatherV2:output:05positional_embedding/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.positional_embedding/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
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
:Ð
,positional_embedding/dense_2/Tensordot/stackPack4positional_embedding/dense_2/Tensordot/Prod:output:06positional_embedding/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
0positional_embedding/dense_2/Tensordot/transpose	Transposex6positional_embedding/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
.positional_embedding/dense_2/Tensordot/ReshapeReshape4positional_embedding/dense_2/Tensordot/transpose:y:05positional_embedding/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿâ
-positional_embedding/dense_2/Tensordot/MatMulMatMul7positional_embedding/dense_2/Tensordot/Reshape:output:0=positional_embedding/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
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
:Û
&positional_embedding/dense_2/TensordotReshape7positional_embedding/dense_2/Tensordot/MatMul:product:08positional_embedding/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
3positional_embedding/dense_2/BiasAdd/ReadVariableOpReadVariableOp<positional_embedding_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
$positional_embedding/dense_2/BiasAddBiasAdd/positional_embedding/dense_2/Tensordot:output:0;positional_embedding/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
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
: ¤
positional_embedding/mulMul-positional_embedding/dense_2/BiasAdd:output:0positional_embedding/Sqrt:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
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
value	B : ó
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
value	B :û
,positional_embedding/strided_slice_1/stack_2Pack7positional_embedding/strided_slice_1/stack_2/0:output:0%positional_embedding/Const_1:output:07positional_embedding/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:æ
$positional_embedding/strided_slice_1StridedSlicepositional_embedding_3449603positional_embedding/strided_slice_1/stack:output:05positional_embedding/strided_slice_1/stack_1:output:05positional_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_mask¥
positional_embedding/addAddV2positional_embedding/mul:z:0-positional_embedding/strided_slice_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
6transformer_encoder/multi_head_attention/einsum/EinsumEinsumpositional_embedding/add:z:0Mtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NI,HIO->...NHOÞ
Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHOÞ
Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_2/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
: ê
0transformer_encoder/multi_head_attention/truedivRealDiv?transformer_encoder/multi_head_attention/einsum/Einsum:output:01transformer_encoder/multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
8transformer_encoder/multi_head_attention/einsum_3/EinsumEinsum4transformer_encoder/multi_head_attention/truediv:z:0Atransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...NHO,...MHO->...HNM¸
0transformer_encoder/multi_head_attention/SoftmaxSoftmaxAtransformer_encoder/multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@transformer_encoder/multi_head_attention/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
>transformer_encoder/multi_head_attention/dropout_1/dropout/MulMul:transformer_encoder/multi_head_attention/Softmax:softmax:0Itransformer_encoder/multi_head_attention/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
@transformer_encoder/multi_head_attention/dropout_1/dropout/ShapeShape:transformer_encoder/multi_head_attention/Softmax:softmax:0*
T0*
_output_shapes
:ú
Wtransformer_encoder/multi_head_attention/dropout_1/dropout/random_uniform/RandomUniformRandomUniformItransformer_encoder/multi_head_attention/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0
Itransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ç
Gtransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqualGreaterEqual`transformer_encoder/multi_head_attention/dropout_1/dropout/random_uniform/RandomUniform:output:0Rtransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
?transformer_encoder/multi_head_attention/dropout_1/dropout/CastCastKtransformer_encoder/multi_head_attention/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@transformer_encoder/multi_head_attention/dropout_1/dropout/Mul_1MulBtransformer_encoder/multi_head_attention/dropout_1/dropout/Mul:z:0Ctransformer_encoder/multi_head_attention/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
8transformer_encoder/multi_head_attention/einsum_4/EinsumEinsumDtransformer_encoder/multi_head_attention/dropout_1/dropout/Mul_1:z:0Atransformer_encoder/multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...HNM,...MHI->...NHIÞ
Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0·
8transformer_encoder/multi_head_attention/einsum_5/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_4/Einsum:output:0Otransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NHI,HIO->...NO½
;transformer_encoder/multi_head_attention/add/ReadVariableOpReadVariableOpDtransformer_encoder_multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0ô
,transformer_encoder/multi_head_attention/addAddV2Atransformer_encoder/multi_head_attention/einsum_5/Einsum:output:0Ctransformer_encoder/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+transformer_encoder/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?Ï
)transformer_encoder/dropout_2/dropout/MulMul0transformer_encoder/multi_head_attention/add:z:04transformer_encoder/dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+transformer_encoder/dropout_2/dropout/ShapeShape0transformer_encoder/multi_head_attention/add:z:0*
T0*
_output_shapes
:Í
Btransformer_encoder/dropout_2/dropout/random_uniform/RandomUniformRandomUniform4transformer_encoder/dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ°
*transformer_encoder/dropout_2/dropout/CastCast6transformer_encoder/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
+transformer_encoder/dropout_2/dropout/Mul_1Mul-transformer_encoder/dropout_2/dropout/Mul:z:0.transformer_encoder/dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ftransformer_encoder/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
4transformer_encoder/layer_normalization/moments/meanMean/transformer_encoder/dropout_2/dropout/Mul_1:z:0Otransformer_encoder/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Á
<transformer_encoder/layer_normalization/moments/StopGradientStopGradient=transformer_encoder/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Atransformer_encoder/layer_normalization/moments/SquaredDifferenceSquaredDifference/transformer_encoder/dropout_2/dropout/Mul_1:z:0Etransformer_encoder/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jtransformer_encoder/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:£
8transformer_encoder/layer_normalization/moments/varianceMeanEtransformer_encoder/layer_normalization/moments/SquaredDifference:z:0Stransformer_encoder/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(|
7transformer_encoder/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ù
5transformer_encoder/layer_normalization/batchnorm/addAddV2Atransformer_encoder/layer_normalization/moments/variance:output:0@transformer_encoder/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
7transformer_encoder/layer_normalization/batchnorm/RsqrtRsqrt9transformer_encoder/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0þ
5transformer_encoder/layer_normalization/batchnorm/mulMul;transformer_encoder/layer_normalization/batchnorm/Rsqrt:y:0Ltransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
7transformer_encoder/layer_normalization/batchnorm/mul_1Mul/transformer_encoder/dropout_2/dropout/Mul_1:z:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_encoder/layer_normalization/batchnorm/mul_2Mul=transformer_encoder/layer_normalization/moments/mean:output:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
@transformer_encoder/layer_normalization/batchnorm/ReadVariableOpReadVariableOpItransformer_encoder_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ú
5transformer_encoder/layer_normalization/batchnorm/subSubHtransformer_encoder/layer_normalization/batchnorm/ReadVariableOp:value:0;transformer_encoder/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_encoder/layer_normalization/batchnorm/add_1AddV2;transformer_encoder/layer_normalization/batchnorm/mul_1:z:09transformer_encoder/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
transformer_encoder/addAddV2;transformer_encoder/layer_normalization/batchnorm/add_1:z:0positional_embedding/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
0transformer_encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÍ
,transformer_encoder/conv1d/Conv1D/ExpandDims
ExpandDimstransformer_encoder/add:z:09transformer_encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpFtransformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0t
2transformer_encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ó
.transformer_encoder/conv1d/Conv1D/ExpandDims_1
ExpandDimsEtransformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0;transformer_encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ÿ
!transformer_encoder/conv1d/Conv1DConv2D5transformer_encoder/conv1d/Conv1D/ExpandDims:output:07transformer_encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
·
)transformer_encoder/conv1d/Conv1D/SqueezeSqueeze*transformer_encoder/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ©
1transformer_encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp:transformer_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
"transformer_encoder/conv1d/BiasAddBiasAdd2transformer_encoder/conv1d/Conv1D/Squeeze:output:09transformer_encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
transformer_encoder/conv1d/ReluRelu+transformer_encoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+transformer_encoder/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?Ì
)transformer_encoder/dropout_3/dropout/MulMul-transformer_encoder/conv1d/Relu:activations:04transformer_encoder/dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+transformer_encoder/dropout_3/dropout/ShapeShape-transformer_encoder/conv1d/Relu:activations:0*
T0*
_output_shapes
:Í
Btransformer_encoder/dropout_3/dropout/random_uniform/RandomUniformRandomUniform4transformer_encoder/dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ°
*transformer_encoder/dropout_3/dropout/CastCast6transformer_encoder/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
+transformer_encoder/dropout_3/dropout/Mul_1Mul-transformer_encoder/dropout_3/dropout/Mul:z:0.transformer_encoder/dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
2transformer_encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿå
.transformer_encoder/conv1d_1/Conv1D/ExpandDims
ExpandDims/transformer_encoder/dropout_3/dropout/Mul_1:z:0;transformer_encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpHtransformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0v
4transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ù
0transformer_encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDimsGtransformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0=transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:
#transformer_encoder/conv1d_1/Conv1DConv2D7transformer_encoder/conv1d_1/Conv1D/ExpandDims:output:09transformer_encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
»
+transformer_encoder/conv1d_1/Conv1D/SqueezeSqueeze,transformer_encoder/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ­
3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ù
$transformer_encoder/conv1d_1/BiasAddBiasAdd4transformer_encoder/conv1d_1/Conv1D/Squeeze:output:0;transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Htransformer_encoder/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
6transformer_encoder/layer_normalization_1/moments/meanMean-transformer_encoder/conv1d_1/BiasAdd:output:0Qtransformer_encoder/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Å
>transformer_encoder/layer_normalization_1/moments/StopGradientStopGradient?transformer_encoder/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ctransformer_encoder/layer_normalization_1/moments/SquaredDifferenceSquaredDifference-transformer_encoder/conv1d_1/BiasAdd:output:0Gtransformer_encoder/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ltransformer_encoder/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:©
:transformer_encoder/layer_normalization_1/moments/varianceMeanGtransformer_encoder/layer_normalization_1/moments/SquaredDifference:z:0Utransformer_encoder/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(~
9transformer_encoder/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ÿ
7transformer_encoder/layer_normalization_1/batchnorm/addAddV2Ctransformer_encoder/layer_normalization_1/moments/variance:output:0Btransformer_encoder/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
9transformer_encoder/layer_normalization_1/batchnorm/RsqrtRsqrt;transformer_encoder/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/mulMul=transformer_encoder/layer_normalization_1/batchnorm/Rsqrt:y:0Ntransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
9transformer_encoder/layer_normalization_1/batchnorm/mul_1Mul-transformer_encoder/conv1d_1/BiasAdd:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
9transformer_encoder/layer_normalization_1/batchnorm/mul_2Mul?transformer_encoder/layer_normalization_1/moments/mean:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpKtransformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/subSubJtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp:value:0=transformer_encoder/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
9transformer_encoder/layer_normalization_1/batchnorm/add_1AddV2=transformer_encoder/layer_normalization_1/batchnorm/mul_1:z:0;transformer_encoder/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
transformer_encoder/add_1AddV2transformer_encoder/add:z:0=transformer_encoder/layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :±
global_average_pooling1d/MeanMeantransformer_encoder/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¿
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp4^positional_embedding/dense_2/BiasAdd/ReadVariableOp6^positional_embedding/dense_2/Tensordot/ReadVariableOp2^transformer_encoder/conv1d/BiasAdd/ReadVariableOp>^transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4^transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp@^transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpA^transformer_encoder/layer_normalization/batchnorm/ReadVariableOpE^transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpC^transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpG^transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp<^transformer_encoder/multi_head_attention/add/ReadVariableOpF^transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:


¤
$__inference_signature_wrapper_344676
input_1
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
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
identity¢StatefulPartitionedCall¬
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
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_343825o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:&"
 
_output_shapes
:

ö

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_344297

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢'multi_head_attention/add/ReadVariableOp¢1multi_head_attention/einsum/Einsum/ReadVariableOp¢3multi_head_attention/einsum_1/Einsum/ReadVariableOp¢3multi_head_attention/einsum_2/Einsum/ReadVariableOp¢3multi_head_attention/einsum_5/Einsum/ReadVariableOp²
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ô
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NI,HIO->...NHO¶
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHO¶
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
: ®
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,multi_head_attention/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?Ê
*multi_head_attention/dropout_1/dropout/MulMul&multi_head_attention/Softmax:softmax:05multi_head_attention/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,multi_head_attention/dropout_1/dropout/ShapeShape&multi_head_attention/Softmax:softmax:0*
T0*
_output_shapes
:Ò
Cmulti_head_attention/dropout_1/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿµ
+multi_head_attention/dropout_1/dropout/CastCast7multi_head_attention/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
,multi_head_attention/dropout_1/dropout/Mul_1Mul.multi_head_attention/dropout_1/dropout/Mul:z:0/multi_head_attention/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/dropout/Mul_1:z:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...HNM,...MHI->...NHI¶
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0û
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0¸
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_2/dropout/MulMulmulti_head_attention/add:z:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dropout_2/dropout/ShapeShapemulti_head_attention/add:z:0*
T0*
_output_shapes
:¥
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>É
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:É
 layer_normalization/moments/meanMeandropout_2/dropout/Mul_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/dropout/Mul_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ç
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75½
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
#layer_normalization/batchnorm/mul_1Muldropout_2/dropout/Mul_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ·
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:Ã
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_3/dropout/MulMulconv1d/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_3/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:¥
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>É
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/dropout/Mul_1:z:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:É
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ë
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²|

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345353

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢'multi_head_attention/add/ReadVariableOp¢1multi_head_attention/einsum/Einsum/ReadVariableOp¢3multi_head_attention/einsum_1/Einsum/ReadVariableOp¢3multi_head_attention/einsum_2/Einsum/ReadVariableOp¢3multi_head_attention/einsum_5/Einsum/ReadVariableOp²
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ô
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NI,HIO->...NHO¶
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHO¶
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
: ®
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'multi_head_attention/dropout_1/IdentityIdentity&multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/Identity:output:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...HNM,...MHI->...NHI¶
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0û
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0¸
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_2/IdentityIdentitymulti_head_attention/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:É
 layer_normalization/moments/meanMeandropout_2/Identity:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/Identity:output:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ç
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75½
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
#layer_normalization/batchnorm/mul_1Muldropout_2/Identity:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ·
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:Ã
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout_3/IdentityIdentityconv1d/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/Identity:output:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:É
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ë
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

&__inference_dense_layer_call_fn_345472

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_344036p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ¬
ð<
"__inference__traced_restore_345952
file_prefix[
Hassignvariableop_transformer_model_3_positional_embedding_dense_2_kernel:	W
Hassignvariableop_1_transformer_model_3_positional_embedding_dense_2_bias:	t
\assignvariableop_2_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel:r
Zassignvariableop_3_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel:t
\assignvariableop_4_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel:y
aassignvariableop_5_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel:n
_assignvariableop_6_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias:	c
Tassignvariableop_7_transformer_model_3_transformer_encoder_layer_normalization_gamma:	b
Sassignvariableop_8_transformer_model_3_transformer_encoder_layer_normalization_beta:	`
Hassignvariableop_9_transformer_model_3_transformer_encoder_conv1d_kernel:V
Gassignvariableop_10_transformer_model_3_transformer_encoder_conv1d_bias:	c
Kassignvariableop_11_transformer_model_3_transformer_encoder_conv1d_1_kernel:X
Iassignvariableop_12_transformer_model_3_transformer_encoder_conv1d_1_bias:	f
Wassignvariableop_13_transformer_model_3_transformer_encoder_layer_normalization_1_gamma:	e
Vassignvariableop_14_transformer_model_3_transformer_encoder_layer_normalization_1_beta:	H
4assignvariableop_15_transformer_model_3_dense_kernel:
A
2assignvariableop_16_transformer_model_3_dense_bias:	I
6assignvariableop_17_transformer_model_3_dense_1_kernel:	B
4assignvariableop_18_transformer_model_3_dense_1_bias:$
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
assignvariableop_29_count: e
Rassignvariableop_30_adam_transformer_model_3_positional_embedding_dense_2_kernel_m:	_
Passignvariableop_31_adam_transformer_model_3_positional_embedding_dense_2_bias_m:	|
dassignvariableop_32_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_m:z
bassignvariableop_33_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_m:|
dassignvariableop_34_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_m:
iassignvariableop_35_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_m:v
gassignvariableop_36_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_m:	k
\assignvariableop_37_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_m:	j
[assignvariableop_38_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_m:	h
Passignvariableop_39_adam_transformer_model_3_transformer_encoder_conv1d_kernel_m:]
Nassignvariableop_40_adam_transformer_model_3_transformer_encoder_conv1d_bias_m:	j
Rassignvariableop_41_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_m:_
Passignvariableop_42_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_m:	m
^assignvariableop_43_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_m:	l
]assignvariableop_44_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_m:	O
;assignvariableop_45_adam_transformer_model_3_dense_kernel_m:
H
9assignvariableop_46_adam_transformer_model_3_dense_bias_m:	P
=assignvariableop_47_adam_transformer_model_3_dense_1_kernel_m:	I
;assignvariableop_48_adam_transformer_model_3_dense_1_bias_m:e
Rassignvariableop_49_adam_transformer_model_3_positional_embedding_dense_2_kernel_v:	_
Passignvariableop_50_adam_transformer_model_3_positional_embedding_dense_2_bias_v:	|
dassignvariableop_51_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_v:z
bassignvariableop_52_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_v:|
dassignvariableop_53_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_v:
iassignvariableop_54_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_v:v
gassignvariableop_55_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_v:	k
\assignvariableop_56_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_v:	j
[assignvariableop_57_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_v:	h
Passignvariableop_58_adam_transformer_model_3_transformer_encoder_conv1d_kernel_v:]
Nassignvariableop_59_adam_transformer_model_3_transformer_encoder_conv1d_bias_v:	j
Rassignvariableop_60_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_v:_
Passignvariableop_61_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_v:	m
^assignvariableop_62_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_v:	l
]assignvariableop_63_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_v:	O
;assignvariableop_64_adam_transformer_model_3_dense_kernel_v:
H
9assignvariableop_65_adam_transformer_model_3_dense_bias_v:	P
=assignvariableop_66_adam_transformer_model_3_dense_1_kernel_v:	I
;assignvariableop_67_adam_transformer_model_3_dense_1_bias_v:
identity_69¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9á
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
valueýBúEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHý
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
valueBEB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ª
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*S
dtypesI
G2E	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOpAssignVariableOpHassignvariableop_transformer_model_3_positional_embedding_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_1AssignVariableOpHassignvariableop_1_transformer_model_3_positional_embedding_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_2AssignVariableOp\assignvariableop_2_transformer_model_3_transformer_encoder_multi_head_attention_query_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:É
AssignVariableOp_3AssignVariableOpZassignvariableop_3_transformer_model_3_transformer_encoder_multi_head_attention_key_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_4AssignVariableOp\assignvariableop_4_transformer_model_3_transformer_encoder_multi_head_attention_value_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_5AssignVariableOpaassignvariableop_5_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_6AssignVariableOp_assignvariableop_6_transformer_model_3_transformer_encoder_multi_head_attention_projection_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_7AssignVariableOpTassignvariableop_7_transformer_model_3_transformer_encoder_layer_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_8AssignVariableOpSassignvariableop_8_transformer_model_3_transformer_encoder_layer_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_9AssignVariableOpHassignvariableop_9_transformer_model_3_transformer_encoder_conv1d_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_10AssignVariableOpGassignvariableop_10_transformer_model_3_transformer_encoder_conv1d_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_11AssignVariableOpKassignvariableop_11_transformer_model_3_transformer_encoder_conv1d_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_12AssignVariableOpIassignvariableop_12_transformer_model_3_transformer_encoder_conv1d_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:È
AssignVariableOp_13AssignVariableOpWassignvariableop_13_transformer_model_3_transformer_encoder_layer_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ç
AssignVariableOp_14AssignVariableOpVassignvariableop_14_transformer_model_3_transformer_encoder_layer_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_15AssignVariableOp4assignvariableop_15_transformer_model_3_dense_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_16AssignVariableOp2assignvariableop_16_transformer_model_3_dense_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_17AssignVariableOp6assignvariableop_17_transformer_model_3_dense_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_18AssignVariableOp4assignvariableop_18_transformer_model_3_dense_1_biasIdentity_18:output:0"/device:CPU:0*
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
:Ã
AssignVariableOp_30AssignVariableOpRassignvariableop_30_adam_transformer_model_3_positional_embedding_dense_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_31AssignVariableOpPassignvariableop_31_adam_transformer_model_3_positional_embedding_dense_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_32AssignVariableOpdassignvariableop_32_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_33AssignVariableOpbassignvariableop_33_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_34AssignVariableOpdassignvariableop_34_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ú
AssignVariableOp_35AssignVariableOpiassignvariableop_35_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_36AssignVariableOpgassignvariableop_36_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_37AssignVariableOp\assignvariableop_37_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_38AssignVariableOp[assignvariableop_38_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_39AssignVariableOpPassignvariableop_39_adam_transformer_model_3_transformer_encoder_conv1d_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_40AssignVariableOpNassignvariableop_40_adam_transformer_model_3_transformer_encoder_conv1d_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_41AssignVariableOpRassignvariableop_41_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_42AssignVariableOpPassignvariableop_42_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_43AssignVariableOp^assignvariableop_43_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_44AssignVariableOp]assignvariableop_44_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_45AssignVariableOp;assignvariableop_45_adam_transformer_model_3_dense_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_46AssignVariableOp9assignvariableop_46_adam_transformer_model_3_dense_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_47AssignVariableOp=assignvariableop_47_adam_transformer_model_3_dense_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_48AssignVariableOp;assignvariableop_48_adam_transformer_model_3_dense_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_49AssignVariableOpRassignvariableop_49_adam_transformer_model_3_positional_embedding_dense_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_50AssignVariableOpPassignvariableop_50_adam_transformer_model_3_positional_embedding_dense_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_51AssignVariableOpdassignvariableop_51_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_52AssignVariableOpbassignvariableop_52_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Õ
AssignVariableOp_53AssignVariableOpdassignvariableop_53_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ú
AssignVariableOp_54AssignVariableOpiassignvariableop_54_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ø
AssignVariableOp_55AssignVariableOpgassignvariableop_55_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_56AssignVariableOp\assignvariableop_56_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ì
AssignVariableOp_57AssignVariableOp[assignvariableop_57_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_58AssignVariableOpPassignvariableop_58_adam_transformer_model_3_transformer_encoder_conv1d_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_59AssignVariableOpNassignvariableop_59_adam_transformer_model_3_transformer_encoder_conv1d_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_60AssignVariableOpRassignvariableop_60_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_61AssignVariableOpPassignvariableop_61_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_62AssignVariableOp^assignvariableop_62_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_63AssignVariableOp]assignvariableop_63_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_64AssignVariableOp;assignvariableop_64_adam_transformer_model_3_dense_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_65AssignVariableOp9assignvariableop_65_adam_transformer_model_3_dense_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_66AssignVariableOp=assignvariableop_66_adam_transformer_model_3_dense_1_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_67AssignVariableOp;assignvariableop_67_adam_transformer_model_3_dense_1_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 §
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
ù	
b
C__inference_dropout_layer_call_and_return_conditional_losses_345510

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_345182

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
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
a
(__inference_dropout_layer_call_fn_345493

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_344140p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
®
4__inference_transformer_model_3_layer_call_fn_344721
x
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
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
identity¢StatefulPartitionedCallÔ
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
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

­&
ð
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344573
input_1.
positional_embedding_344526:	*
positional_embedding_344528:	
positional_embedding_3445302
transformer_encoder_344533:2
transformer_encoder_344535:2
transformer_encoder_344537:2
transformer_encoder_344539:)
transformer_encoder_344541:	)
transformer_encoder_344543:	)
transformer_encoder_344545:	2
transformer_encoder_344547:)
transformer_encoder_344549:	2
transformer_encoder_344551:)
transformer_encoder_344553:	)
transformer_encoder_344555:	)
transformer_encoder_344557:	 
dense_344561:

dense_344563:	!
dense_1_344567:	
dense_1_344569:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢,positional_embedding/StatefulPartitionedCall¢+transformer_encoder/StatefulPartitionedCallÇ
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1positional_embedding_344526positional_embedding_344528positional_embedding_344530*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_343899
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_344533transformer_encoder_344535transformer_encoder_344537transformer_encoder_344539transformer_encoder_344541transformer_encoder_344543transformer_encoder_344545transformer_encoder_344547transformer_encoder_344549transformer_encoder_344551transformer_encoder_344553transformer_encoder_344555transformer_encoder_344557*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_343996
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_343835
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_344561dense_344563*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_344036Ú
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_344047
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_344567dense_1_344569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_344060w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:&"
 
_output_shapes
:

¢
D
(__inference_dropout_layer_call_fn_345488

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_344047a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4
±
P__inference_positional_embedding_layer_call_and_return_conditional_losses_345171
x<
)dense_2_tensordot_readvariableop_resource:	6
'dense_2_biasadd_readvariableop_resource:	
unknown
identity¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp6
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
valueB:Ñ
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
value	B : Û
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
value	B : ß
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
value	B : ¼
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
:ÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
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
:ÿÿÿÿÿÿÿÿÿG
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
value	B : ­
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
value	B :§
strided_slice_1/stack_2Pack"strided_slice_1/stack_2/0:output:0Const_1:output:0"strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:þ
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
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : :
2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

Í'
	
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344623
input_1.
positional_embedding_344576:	*
positional_embedding_344578:	
positional_embedding_3445802
transformer_encoder_344583:2
transformer_encoder_344585:2
transformer_encoder_344587:2
transformer_encoder_344589:)
transformer_encoder_344591:	)
transformer_encoder_344593:	)
transformer_encoder_344595:	2
transformer_encoder_344597:)
transformer_encoder_344599:	2
transformer_encoder_344601:)
transformer_encoder_344603:	)
transformer_encoder_344605:	)
transformer_encoder_344607:	 
dense_344611:

dense_344613:	!
dense_1_344617:	
dense_1_344619:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢,positional_embedding/StatefulPartitionedCall¢+transformer_encoder/StatefulPartitionedCallÇ
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1positional_embedding_344576positional_embedding_344578positional_embedding_344580*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_343899
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_344583transformer_encoder_344585transformer_encoder_344587transformer_encoder_344589transformer_encoder_344591transformer_encoder_344593transformer_encoder_344595transformer_encoder_344597transformer_encoder_344599transformer_encoder_344601transformer_encoder_344603transformer_encoder_344605transformer_encoder_344607*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_344297
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_343835
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_344611dense_344613*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_344036ê
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_344140
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_344617dense_1_344619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_344060w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:&"
 
_output_shapes
:

Ìö
©
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344922
xQ
>positional_embedding_dense_2_tensordot_readvariableop_resource:	K
<positional_embedding_dense_2_biasadd_readvariableop_resource:	
positional_embedding_344804f
Ntransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource:h
Ptransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource:S
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
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢3positional_embedding/dense_2/BiasAdd/ReadVariableOp¢5positional_embedding/dense_2/Tensordot/ReadVariableOp¢1transformer_encoder/conv1d/BiasAdd/ReadVariableOp¢=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp¢?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢@transformer_encoder/layer_normalization/batchnorm/ReadVariableOp¢Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp¢Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp¢Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp¢;transformer_encoder/multi_head_attention/add/ReadVariableOp¢Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp¢Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp¢Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp¢Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpK
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
valueB:º
"positional_embedding/strided_sliceStridedSlice#positional_embedding/Shape:output:01positional_embedding/strided_slice/stack:output:03positional_embedding/strided_slice/stack_1:output:03positional_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskµ
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
value	B : ¯
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
value	B : ³
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
valueB: Å
+positional_embedding/dense_2/Tensordot/ProdProd8positional_embedding/dense_2/Tensordot/GatherV2:output:05positional_embedding/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.positional_embedding/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
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
:Ð
,positional_embedding/dense_2/Tensordot/stackPack4positional_embedding/dense_2/Tensordot/Prod:output:06positional_embedding/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
0positional_embedding/dense_2/Tensordot/transpose	Transposex6positional_embedding/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
.positional_embedding/dense_2/Tensordot/ReshapeReshape4positional_embedding/dense_2/Tensordot/transpose:y:05positional_embedding/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿâ
-positional_embedding/dense_2/Tensordot/MatMulMatMul7positional_embedding/dense_2/Tensordot/Reshape:output:0=positional_embedding/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
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
:Û
&positional_embedding/dense_2/TensordotReshape7positional_embedding/dense_2/Tensordot/MatMul:product:08positional_embedding/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
3positional_embedding/dense_2/BiasAdd/ReadVariableOpReadVariableOp<positional_embedding_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
$positional_embedding/dense_2/BiasAddBiasAdd/positional_embedding/dense_2/Tensordot:output:0;positional_embedding/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
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
: ¤
positional_embedding/mulMul-positional_embedding/dense_2/BiasAdd:output:0positional_embedding/Sqrt:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
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
value	B : ó
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
value	B :û
,positional_embedding/strided_slice_1/stack_2Pack7positional_embedding/strided_slice_1/stack_2/0:output:0%positional_embedding/Const_1:output:07positional_embedding/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:æ
$positional_embedding/strided_slice_1StridedSlicepositional_embedding_3448043positional_embedding/strided_slice_1/stack:output:05positional_embedding/strided_slice_1/stack_1:output:05positional_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_mask¥
positional_embedding/addAddV2positional_embedding/mul:z:0-positional_embedding/strided_slice_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
Etransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOpNtransformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
6transformer_encoder/multi_head_attention/einsum/EinsumEinsumpositional_embedding/add:z:0Mtransformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NI,HIO->...NHOÞ
Gtransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_1/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHOÞ
Gtransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0
8transformer_encoder/multi_head_attention/einsum_2/EinsumEinsumpositional_embedding/add:z:0Otransformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
: ê
0transformer_encoder/multi_head_attention/truedivRealDiv?transformer_encoder/multi_head_attention/einsum/Einsum:output:01transformer_encoder/multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
8transformer_encoder/multi_head_attention/einsum_3/EinsumEinsum4transformer_encoder/multi_head_attention/truediv:z:0Atransformer_encoder/multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...NHO,...MHO->...HNM¸
0transformer_encoder/multi_head_attention/SoftmaxSoftmaxAtransformer_encoder/multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
;transformer_encoder/multi_head_attention/dropout_1/IdentityIdentity:transformer_encoder/multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
8transformer_encoder/multi_head_attention/einsum_4/EinsumEinsumDtransformer_encoder/multi_head_attention/dropout_1/Identity:output:0Atransformer_encoder/multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...HNM,...MHI->...NHIÞ
Gtransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOpPtransformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0·
8transformer_encoder/multi_head_attention/einsum_5/EinsumEinsumAtransformer_encoder/multi_head_attention/einsum_4/Einsum:output:0Otransformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NHI,HIO->...NO½
;transformer_encoder/multi_head_attention/add/ReadVariableOpReadVariableOpDtransformer_encoder_multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0ô
,transformer_encoder/multi_head_attention/addAddV2Atransformer_encoder/multi_head_attention/einsum_5/Einsum:output:0Ctransformer_encoder/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&transformer_encoder/dropout_2/IdentityIdentity0transformer_encoder/multi_head_attention/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ftransformer_encoder/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
4transformer_encoder/layer_normalization/moments/meanMean/transformer_encoder/dropout_2/Identity:output:0Otransformer_encoder/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Á
<transformer_encoder/layer_normalization/moments/StopGradientStopGradient=transformer_encoder/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Atransformer_encoder/layer_normalization/moments/SquaredDifferenceSquaredDifference/transformer_encoder/dropout_2/Identity:output:0Etransformer_encoder/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jtransformer_encoder/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:£
8transformer_encoder/layer_normalization/moments/varianceMeanEtransformer_encoder/layer_normalization/moments/SquaredDifference:z:0Stransformer_encoder/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(|
7transformer_encoder/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ù
5transformer_encoder/layer_normalization/batchnorm/addAddV2Atransformer_encoder/layer_normalization/moments/variance:output:0@transformer_encoder/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
7transformer_encoder/layer_normalization/batchnorm/RsqrtRsqrt9transformer_encoder/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Dtransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0þ
5transformer_encoder/layer_normalization/batchnorm/mulMul;transformer_encoder/layer_normalization/batchnorm/Rsqrt:y:0Ltransformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
7transformer_encoder/layer_normalization/batchnorm/mul_1Mul/transformer_encoder/dropout_2/Identity:output:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_encoder/layer_normalization/batchnorm/mul_2Mul=transformer_encoder/layer_normalization/moments/mean:output:09transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
@transformer_encoder/layer_normalization/batchnorm/ReadVariableOpReadVariableOpItransformer_encoder_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0ú
5transformer_encoder/layer_normalization/batchnorm/subSubHtransformer_encoder/layer_normalization/batchnorm/ReadVariableOp:value:0;transformer_encoder/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
7transformer_encoder/layer_normalization/batchnorm/add_1AddV2;transformer_encoder/layer_normalization/batchnorm/mul_1:z:09transformer_encoder/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
transformer_encoder/addAddV2;transformer_encoder/layer_normalization/batchnorm/add_1:z:0positional_embedding/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
0transformer_encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÍ
,transformer_encoder/conv1d/Conv1D/ExpandDims
ExpandDimstransformer_encoder/add:z:09transformer_encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
=transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpFtransformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0t
2transformer_encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ó
.transformer_encoder/conv1d/Conv1D/ExpandDims_1
ExpandDimsEtransformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0;transformer_encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:ÿ
!transformer_encoder/conv1d/Conv1DConv2D5transformer_encoder/conv1d/Conv1D/ExpandDims:output:07transformer_encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
·
)transformer_encoder/conv1d/Conv1D/SqueezeSqueeze*transformer_encoder/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ©
1transformer_encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp:transformer_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
"transformer_encoder/conv1d/BiasAddBiasAdd2transformer_encoder/conv1d/Conv1D/Squeeze:output:09transformer_encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
transformer_encoder/conv1d/ReluRelu+transformer_encoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&transformer_encoder/dropout_3/IdentityIdentity-transformer_encoder/conv1d/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
2transformer_encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿå
.transformer_encoder/conv1d_1/Conv1D/ExpandDims
ExpandDims/transformer_encoder/dropout_3/Identity:output:0;transformer_encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
?transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpHtransformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0v
4transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ù
0transformer_encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDimsGtransformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0=transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:
#transformer_encoder/conv1d_1/Conv1DConv2D7transformer_encoder/conv1d_1/Conv1D/ExpandDims:output:09transformer_encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
»
+transformer_encoder/conv1d_1/Conv1D/SqueezeSqueeze,transformer_encoder/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ­
3transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ù
$transformer_encoder/conv1d_1/BiasAddBiasAdd4transformer_encoder/conv1d_1/Conv1D/Squeeze:output:0;transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Htransformer_encoder/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
6transformer_encoder/layer_normalization_1/moments/meanMean-transformer_encoder/conv1d_1/BiasAdd:output:0Qtransformer_encoder/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(Å
>transformer_encoder/layer_normalization_1/moments/StopGradientStopGradient?transformer_encoder/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ctransformer_encoder/layer_normalization_1/moments/SquaredDifferenceSquaredDifference-transformer_encoder/conv1d_1/BiasAdd:output:0Gtransformer_encoder/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ltransformer_encoder/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:©
:transformer_encoder/layer_normalization_1/moments/varianceMeanGtransformer_encoder/layer_normalization_1/moments/SquaredDifference:z:0Utransformer_encoder/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(~
9transformer_encoder/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75ÿ
7transformer_encoder/layer_normalization_1/batchnorm/addAddV2Ctransformer_encoder/layer_normalization_1/moments/variance:output:0Btransformer_encoder/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
9transformer_encoder/layer_normalization_1/batchnorm/RsqrtRsqrt;transformer_encoder/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
Ftransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/mulMul=transformer_encoder/layer_normalization_1/batchnorm/Rsqrt:y:0Ntransformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
9transformer_encoder/layer_normalization_1/batchnorm/mul_1Mul-transformer_encoder/conv1d_1/BiasAdd:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
9transformer_encoder/layer_normalization_1/batchnorm/mul_2Mul?transformer_encoder/layer_normalization_1/moments/mean:output:0;transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
Btransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpKtransformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
7transformer_encoder/layer_normalization_1/batchnorm/subSubJtransformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp:value:0=transformer_encoder/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
9transformer_encoder/layer_normalization_1/batchnorm/add_1AddV2=transformer_encoder/layer_normalization_1/batchnorm/mul_1:z:0;transformer_encoder/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
transformer_encoder/add_1AddV2transformer_encoder/add:z:0=transformer_encoder/layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :±
global_average_pooling1d/MeanMeantransformer_encoder/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMul&global_average_pooling1d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp4^positional_embedding/dense_2/BiasAdd/ReadVariableOp6^positional_embedding/dense_2/Tensordot/ReadVariableOp2^transformer_encoder/conv1d/BiasAdd/ReadVariableOp>^transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4^transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp@^transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpA^transformer_encoder/layer_normalization/batchnorm/ReadVariableOpE^transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpC^transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpG^transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp<^transformer_encoder/multi_head_attention/add/ReadVariableOpF^transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpH^transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
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
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

ê
õ
4__inference_transformer_encoder_layer_call_fn_345233

inputs
unknown:!
	unknown_0:!
	unknown_1:!
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	!
	unknown_6:
	unknown_7:	!
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	
identity¢StatefulPartitionedCallý
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
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_343996t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

õ
A__inference_dense_layer_call_and_return_conditional_losses_345483

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£

õ
C__inference_dense_1_layer_call_and_return_conditional_losses_345202

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª´

!__inference__wrapped_model_343825
input_1e
Rtransformer_model_3_positional_embedding_dense_2_tensordot_readvariableop_resource:	_
Ptransformer_model_3_positional_embedding_dense_2_biasadd_readvariableop_resource:	3
/transformer_model_3_positional_embedding_343707z
btransformer_model_3_transformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource:|
dtransformer_model_3_transformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource:|
dtransformer_model_3_transformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource:|
dtransformer_model_3_transformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource:g
Xtransformer_model_3_transformer_encoder_multi_head_attention_add_readvariableop_resource:	p
atransformer_model_3_transformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource:	l
]transformer_model_3_transformer_encoder_layer_normalization_batchnorm_readvariableop_resource:	r
Ztransformer_model_3_transformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource:]
Ntransformer_model_3_transformer_encoder_conv1d_biasadd_readvariableop_resource:	t
\transformer_model_3_transformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource:_
Ptransformer_model_3_transformer_encoder_conv1d_1_biasadd_readvariableop_resource:	r
ctransformer_model_3_transformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource:	n
_transformer_model_3_transformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource:	L
8transformer_model_3_dense_matmul_readvariableop_resource:
H
9transformer_model_3_dense_biasadd_readvariableop_resource:	M
:transformer_model_3_dense_1_matmul_readvariableop_resource:	I
;transformer_model_3_dense_1_biasadd_readvariableop_resource:
identity¢0transformer_model_3/dense/BiasAdd/ReadVariableOp¢/transformer_model_3/dense/MatMul/ReadVariableOp¢2transformer_model_3/dense_1/BiasAdd/ReadVariableOp¢1transformer_model_3/dense_1/MatMul/ReadVariableOp¢Gtransformer_model_3/positional_embedding/dense_2/BiasAdd/ReadVariableOp¢Itransformer_model_3/positional_embedding/dense_2/Tensordot/ReadVariableOp¢Etransformer_model_3/transformer_encoder/conv1d/BiasAdd/ReadVariableOp¢Qtransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢Gtransformer_model_3/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp¢Stransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢Ttransformer_model_3/transformer_encoder/layer_normalization/batchnorm/ReadVariableOp¢Xtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp¢Vtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp¢Ztransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp¢Otransformer_model_3/transformer_encoder/multi_head_attention/add/ReadVariableOp¢Ytransformer_model_3/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp¢[transformer_model_3/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp¢[transformer_model_3/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp¢[transformer_model_3/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpe
.transformer_model_3/positional_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:
<transformer_model_3/positional_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
>transformer_model_3/positional_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>transformer_model_3/positional_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6transformer_model_3/positional_embedding/strided_sliceStridedSlice7transformer_model_3/positional_embedding/Shape:output:0Etransformer_model_3/positional_embedding/strided_slice/stack:output:0Gtransformer_model_3/positional_embedding/strided_slice/stack_1:output:0Gtransformer_model_3/positional_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÝ
Itransformer_model_3/positional_embedding/dense_2/Tensordot/ReadVariableOpReadVariableOpRtransformer_model_3_positional_embedding_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0
?transformer_model_3/positional_embedding/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
?transformer_model_3/positional_embedding/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       w
@transformer_model_3/positional_embedding/dense_2/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:
Htransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Ctransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2GatherV2Itransformer_model_3/positional_embedding/dense_2/Tensordot/Shape:output:0Htransformer_model_3/positional_embedding/dense_2/Tensordot/free:output:0Qtransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Jtransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Etransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2_1GatherV2Itransformer_model_3/positional_embedding/dense_2/Tensordot/Shape:output:0Htransformer_model_3/positional_embedding/dense_2/Tensordot/axes:output:0Stransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
@transformer_model_3/positional_embedding/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?transformer_model_3/positional_embedding/dense_2/Tensordot/ProdProdLtransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2:output:0Itransformer_model_3/positional_embedding/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 
Btransformer_model_3/positional_embedding/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Atransformer_model_3/positional_embedding/dense_2/Tensordot/Prod_1ProdNtransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2_1:output:0Ktransformer_model_3/positional_embedding/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Ftransformer_model_3/positional_embedding/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
Atransformer_model_3/positional_embedding/dense_2/Tensordot/concatConcatV2Htransformer_model_3/positional_embedding/dense_2/Tensordot/free:output:0Htransformer_model_3/positional_embedding/dense_2/Tensordot/axes:output:0Otransformer_model_3/positional_embedding/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
@transformer_model_3/positional_embedding/dense_2/Tensordot/stackPackHtransformer_model_3/positional_embedding/dense_2/Tensordot/Prod:output:0Jtransformer_model_3/positional_embedding/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ü
Dtransformer_model_3/positional_embedding/dense_2/Tensordot/transpose	Transposeinput_1Jtransformer_model_3/positional_embedding/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Btransformer_model_3/positional_embedding/dense_2/Tensordot/ReshapeReshapeHtransformer_model_3/positional_embedding/dense_2/Tensordot/transpose:y:0Itransformer_model_3/positional_embedding/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Atransformer_model_3/positional_embedding/dense_2/Tensordot/MatMulMatMulKtransformer_model_3/positional_embedding/dense_2/Tensordot/Reshape:output:0Qtransformer_model_3/positional_embedding/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Btransformer_model_3/positional_embedding/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
Htransformer_model_3/positional_embedding/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
Ctransformer_model_3/positional_embedding/dense_2/Tensordot/concat_1ConcatV2Ltransformer_model_3/positional_embedding/dense_2/Tensordot/GatherV2:output:0Ktransformer_model_3/positional_embedding/dense_2/Tensordot/Const_2:output:0Qtransformer_model_3/positional_embedding/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
:transformer_model_3/positional_embedding/dense_2/TensordotReshapeKtransformer_model_3/positional_embedding/dense_2/Tensordot/MatMul:product:0Ltransformer_model_3/positional_embedding/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
Gtransformer_model_3/positional_embedding/dense_2/BiasAdd/ReadVariableOpReadVariableOpPtransformer_model_3_positional_embedding_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
8transformer_model_3/positional_embedding/dense_2/BiasAddBiasAddCtransformer_model_3/positional_embedding/dense_2/Tensordot:output:0Otransformer_model_3/positional_embedding/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
/transformer_model_3/positional_embedding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :
-transformer_model_3/positional_embedding/CastCast8transformer_model_3/positional_embedding/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
-transformer_model_3/positional_embedding/SqrtSqrt1transformer_model_3/positional_embedding/Cast:y:0*
T0*
_output_shapes
: à
,transformer_model_3/positional_embedding/mulMulAtransformer_model_3/positional_embedding/dense_2/BiasAdd:output:01transformer_model_3/positional_embedding/Sqrt:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.transformer_model_3/positional_embedding/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
0transformer_model_3/positional_embedding/Const_1Const*
_output_shapes
: *
dtype0*
value	B :
@transformer_model_3/positional_embedding/strided_slice_1/stack/0Const*
_output_shapes
: *
dtype0*
value	B : 
@transformer_model_3/positional_embedding/strided_slice_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B : Ã
>transformer_model_3/positional_embedding/strided_slice_1/stackPackItransformer_model_3/positional_embedding/strided_slice_1/stack/0:output:07transformer_model_3/positional_embedding/Const:output:0Itransformer_model_3/positional_embedding/strided_slice_1/stack/2:output:0*
N*
T0*
_output_shapes
:
Btransformer_model_3/positional_embedding/strided_slice_1/stack_1/0Const*
_output_shapes
: *
dtype0*
value	B : 
Btransformer_model_3/positional_embedding/strided_slice_1/stack_1/2Const*
_output_shapes
: *
dtype0*
value	B : Ñ
@transformer_model_3/positional_embedding/strided_slice_1/stack_1PackKtransformer_model_3/positional_embedding/strided_slice_1/stack_1/0:output:0?transformer_model_3/positional_embedding/strided_slice:output:0Ktransformer_model_3/positional_embedding/strided_slice_1/stack_1/2:output:0*
N*
T0*
_output_shapes
:
Btransformer_model_3/positional_embedding/strided_slice_1/stack_2/0Const*
_output_shapes
: *
dtype0*
value	B :
Btransformer_model_3/positional_embedding/strided_slice_1/stack_2/2Const*
_output_shapes
: *
dtype0*
value	B :Ë
@transformer_model_3/positional_embedding/strided_slice_1/stack_2PackKtransformer_model_3/positional_embedding/strided_slice_1/stack_2/0:output:09transformer_model_3/positional_embedding/Const_1:output:0Ktransformer_model_3/positional_embedding/strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:Ê
8transformer_model_3/positional_embedding/strided_slice_1StridedSlice/transformer_model_3_positional_embedding_343707Gtransformer_model_3/positional_embedding/strided_slice_1/stack:output:0Itransformer_model_3/positional_embedding/strided_slice_1/stack_1:output:0Itransformer_model_3/positional_embedding/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:*

begin_mask*
end_mask*
new_axis_maská
,transformer_model_3/positional_embedding/addAddV20transformer_model_3/positional_embedding/mul:z:0Atransformer_model_3/positional_embedding/strided_slice_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ytransformer_model_3/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_model_3_transformer_encoder_multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Î
Jtransformer_model_3/transformer_encoder/multi_head_attention/einsum/EinsumEinsum0transformer_model_3/positional_embedding/add:z:0atransformer_model_3/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NI,HIO->...NHO
[transformer_model_3/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOpdtransformer_model_3_transformer_encoder_multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ò
Ltransformer_model_3/transformer_encoder/multi_head_attention/einsum_1/EinsumEinsum0transformer_model_3/positional_embedding/add:z:0ctransformer_model_3/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHO
[transformer_model_3/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOpdtransformer_model_3_transformer_encoder_multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ò
Ltransformer_model_3/transformer_encoder/multi_head_attention/einsum_2/EinsumEinsum0transformer_model_3/positional_embedding/add:z:0ctransformer_model_3/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHO
Btransformer_model_3/transformer_encoder/multi_head_attention/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  C·
Atransformer_model_3/transformer_encoder/multi_head_attention/SqrtSqrtKtransformer_model_3/transformer_encoder/multi_head_attention/Const:output:0*
T0*
_output_shapes
: ¦
Dtransformer_model_3/transformer_encoder/multi_head_attention/truedivRealDivStransformer_model_3/transformer_encoder/multi_head_attention/einsum/Einsum:output:0Etransformer_model_3/transformer_encoder/multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
Ltransformer_model_3/transformer_encoder/multi_head_attention/einsum_3/EinsumEinsumHtransformer_model_3/transformer_encoder/multi_head_attention/truediv:z:0Utransformer_model_3/transformer_encoder/multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...NHO,...MHO->...HNMà
Dtransformer_model_3/transformer_encoder/multi_head_attention/SoftmaxSoftmaxUtransformer_model_3/transformer_encoder/multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
Otransformer_model_3/transformer_encoder/multi_head_attention/dropout_1/IdentityIdentityNtransformer_model_3/transformer_encoder/multi_head_attention/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
Ltransformer_model_3/transformer_encoder/multi_head_attention/einsum_4/EinsumEinsumXtransformer_model_3/transformer_encoder/multi_head_attention/dropout_1/Identity:output:0Utransformer_model_3/transformer_encoder/multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...HNM,...MHI->...NHI
[transformer_model_3/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOpdtransformer_model_3_transformer_encoder_multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0ó
Ltransformer_model_3/transformer_encoder/multi_head_attention/einsum_5/EinsumEinsumUtransformer_model_3/transformer_encoder/multi_head_attention/einsum_4/Einsum:output:0ctransformer_model_3/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NHI,HIO->...NOå
Otransformer_model_3/transformer_encoder/multi_head_attention/add/ReadVariableOpReadVariableOpXtransformer_model_3_transformer_encoder_multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0°
@transformer_model_3/transformer_encoder/multi_head_attention/addAddV2Utransformer_model_3/transformer_encoder/multi_head_attention/einsum_5/Einsum:output:0Wtransformer_model_3/transformer_encoder/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
:transformer_model_3/transformer_encoder/dropout_2/IdentityIdentityDtransformer_model_3/transformer_encoder/multi_head_attention/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
Ztransformer_model_3/transformer_encoder/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Á
Htransformer_model_3/transformer_encoder/layer_normalization/moments/meanMeanCtransformer_model_3/transformer_encoder/dropout_2/Identity:output:0ctransformer_model_3/transformer_encoder/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(é
Ptransformer_model_3/transformer_encoder/layer_normalization/moments/StopGradientStopGradientQtransformer_model_3/transformer_encoder/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
Utransformer_model_3/transformer_encoder/layer_normalization/moments/SquaredDifferenceSquaredDifferenceCtransformer_model_3/transformer_encoder/dropout_2/Identity:output:0Ytransformer_model_3/transformer_encoder/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
^transformer_model_3/transformer_encoder/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ß
Ltransformer_model_3/transformer_encoder/layer_normalization/moments/varianceMeanYtransformer_model_3/transformer_encoder/layer_normalization/moments/SquaredDifference:z:0gtransformer_model_3/transformer_encoder/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
Ktransformer_model_3/transformer_encoder/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75µ
Itransformer_model_3/transformer_encoder/layer_normalization/batchnorm/addAddV2Utransformer_model_3/transformer_encoder/layer_normalization/moments/variance:output:0Ttransformer_model_3/transformer_encoder/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
Ktransformer_model_3/transformer_encoder/layer_normalization/batchnorm/RsqrtRsqrtMtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
Xtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpatransformer_model_3_transformer_encoder_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
Itransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mulMulOtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/Rsqrt:y:0`transformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ktransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul_1MulCtransformer_model_3/transformer_encoder/dropout_2/Identity:output:0Mtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
Ktransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul_2MulQtransformer_model_3/transformer_encoder/layer_normalization/moments/mean:output:0Mtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
Ttransformer_model_3/transformer_encoder/layer_normalization/batchnorm/ReadVariableOpReadVariableOp]transformer_model_3_transformer_encoder_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¶
Itransformer_model_3/transformer_encoder/layer_normalization/batchnorm/subSub\transformer_model_3/transformer_encoder/layer_normalization/batchnorm/ReadVariableOp:value:0Otransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
Ktransformer_model_3/transformer_encoder/layer_normalization/batchnorm/add_1AddV2Otransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul_1:z:0Mtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
+transformer_model_3/transformer_encoder/addAddV2Otransformer_model_3/transformer_encoder/layer_normalization/batchnorm/add_1:z:00transformer_model_3/positional_embedding/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dtransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
@transformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims
ExpandDims/transformer_model_3/transformer_encoder/add:z:0Mtransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
Qtransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpZtransformer_model_3_transformer_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0
Ftransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¯
Btransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1
ExpandDimsYtransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0Otransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:»
5transformer_model_3/transformer_encoder/conv1d/Conv1DConv2DItransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims:output:0Ktransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
ß
=transformer_model_3/transformer_encoder/conv1d/Conv1D/SqueezeSqueeze>transformer_model_3/transformer_encoder/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿÑ
Etransformer_model_3/transformer_encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOpNtransformer_model_3_transformer_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
6transformer_model_3/transformer_encoder/conv1d/BiasAddBiasAddFtransformer_model_3/transformer_encoder/conv1d/Conv1D/Squeeze:output:0Mtransformer_model_3/transformer_encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
3transformer_model_3/transformer_encoder/conv1d/ReluRelu?transformer_model_3/transformer_encoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
:transformer_model_3/transformer_encoder/dropout_3/IdentityIdentityAtransformer_model_3/transformer_encoder/conv1d/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ftransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¡
Btransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims
ExpandDimsCtransformer_model_3/transformer_encoder/dropout_3/Identity:output:0Otransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
Stransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp\transformer_model_3_transformer_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0
Htransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : µ
Dtransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1
ExpandDims[transformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Qtransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:Á
7transformer_model_3/transformer_encoder/conv1d_1/Conv1DConv2DKtransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims:output:0Mtransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
ã
?transformer_model_3/transformer_encoder/conv1d_1/Conv1D/SqueezeSqueeze@transformer_model_3/transformer_encoder/conv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿÕ
Gtransformer_model_3/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOpPtransformer_model_3_transformer_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
8transformer_model_3/transformer_encoder/conv1d_1/BiasAddBiasAddHtransformer_model_3/transformer_encoder/conv1d_1/Conv1D/Squeeze:output:0Otransformer_model_3/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
\transformer_model_3/transformer_encoder/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ã
Jtransformer_model_3/transformer_encoder/layer_normalization_1/moments/meanMeanAtransformer_model_3/transformer_encoder/conv1d_1/BiasAdd:output:0etransformer_model_3/transformer_encoder/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(í
Rtransformer_model_3/transformer_encoder/layer_normalization_1/moments/StopGradientStopGradientStransformer_model_3/transformer_encoder/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Wtransformer_model_3/transformer_encoder/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceAtransformer_model_3/transformer_encoder/conv1d_1/BiasAdd:output:0[transformer_model_3/transformer_encoder/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
`transformer_model_3/transformer_encoder/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:å
Ntransformer_model_3/transformer_encoder/layer_normalization_1/moments/varianceMean[transformer_model_3/transformer_encoder/layer_normalization_1/moments/SquaredDifference:z:0itransformer_model_3/transformer_encoder/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
Mtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75»
Ktransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/addAddV2Wtransformer_model_3/transformer_encoder/layer_normalization_1/moments/variance:output:0Vtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
Mtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/RsqrtRsqrtOtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
Ztransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpctransformer_model_3_transformer_encoder_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0À
Ktransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mulMulQtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/Rsqrt:y:0btransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Mtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul_1MulAtransformer_model_3/transformer_encoder/conv1d_1/BiasAdd:output:0Otransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
Mtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul_2MulStransformer_model_3/transformer_encoder/layer_normalization_1/moments/mean:output:0Otransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
Vtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp_transformer_model_3_transformer_encoder_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¼
Ktransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/subSub^transformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp:value:0Qtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
Mtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/add_1AddV2Qtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul_1:z:0Otransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
-transformer_model_3/transformer_encoder/add_1AddV2/transformer_model_3/transformer_encoder/add:z:0Qtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ctransformer_model_3/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :í
1transformer_model_3/global_average_pooling1d/MeanMean1transformer_model_3/transformer_encoder/add_1:z:0Ltransformer_model_3/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
/transformer_model_3/dense/MatMul/ReadVariableOpReadVariableOp8transformer_model_3_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ò
 transformer_model_3/dense/MatMulMatMul:transformer_model_3/global_average_pooling1d/Mean:output:07transformer_model_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0transformer_model_3/dense/BiasAdd/ReadVariableOpReadVariableOp9transformer_model_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Å
!transformer_model_3/dense/BiasAddBiasAdd*transformer_model_3/dense/MatMul:product:08transformer_model_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
transformer_model_3/dense/ReluRelu*transformer_model_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$transformer_model_3/dropout/IdentityIdentity,transformer_model_3/dense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
1transformer_model_3/dense_1/MatMul/ReadVariableOpReadVariableOp:transformer_model_3_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0È
"transformer_model_3/dense_1/MatMulMatMul-transformer_model_3/dropout/Identity:output:09transformer_model_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2transformer_model_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp;transformer_model_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
#transformer_model_3/dense_1/BiasAddBiasAdd,transformer_model_3/dense_1/MatMul:product:0:transformer_model_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#transformer_model_3/dense_1/SoftmaxSoftmax,transformer_model_3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
IdentityIdentity-transformer_model_3/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp1^transformer_model_3/dense/BiasAdd/ReadVariableOp0^transformer_model_3/dense/MatMul/ReadVariableOp3^transformer_model_3/dense_1/BiasAdd/ReadVariableOp2^transformer_model_3/dense_1/MatMul/ReadVariableOpH^transformer_model_3/positional_embedding/dense_2/BiasAdd/ReadVariableOpJ^transformer_model_3/positional_embedding/dense_2/Tensordot/ReadVariableOpF^transformer_model_3/transformer_encoder/conv1d/BiasAdd/ReadVariableOpR^transformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpH^transformer_model_3/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpT^transformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpU^transformer_model_3/transformer_encoder/layer_normalization/batchnorm/ReadVariableOpY^transformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpW^transformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp[^transformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_model_3/transformer_encoder/multi_head_attention/add/ReadVariableOpZ^transformer_model_3/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp\^transformer_model_3/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp\^transformer_model_3/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp\^transformer_model_3/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 2d
0transformer_model_3/dense/BiasAdd/ReadVariableOp0transformer_model_3/dense/BiasAdd/ReadVariableOp2b
/transformer_model_3/dense/MatMul/ReadVariableOp/transformer_model_3/dense/MatMul/ReadVariableOp2h
2transformer_model_3/dense_1/BiasAdd/ReadVariableOp2transformer_model_3/dense_1/BiasAdd/ReadVariableOp2f
1transformer_model_3/dense_1/MatMul/ReadVariableOp1transformer_model_3/dense_1/MatMul/ReadVariableOp2
Gtransformer_model_3/positional_embedding/dense_2/BiasAdd/ReadVariableOpGtransformer_model_3/positional_embedding/dense_2/BiasAdd/ReadVariableOp2
Itransformer_model_3/positional_embedding/dense_2/Tensordot/ReadVariableOpItransformer_model_3/positional_embedding/dense_2/Tensordot/ReadVariableOp2
Etransformer_model_3/transformer_encoder/conv1d/BiasAdd/ReadVariableOpEtransformer_model_3/transformer_encoder/conv1d/BiasAdd/ReadVariableOp2¦
Qtransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOpQtransformer_model_3/transformer_encoder/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2
Gtransformer_model_3/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOpGtransformer_model_3/transformer_encoder/conv1d_1/BiasAdd/ReadVariableOp2ª
Stransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpStransformer_model_3/transformer_encoder/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2¬
Ttransformer_model_3/transformer_encoder/layer_normalization/batchnorm/ReadVariableOpTtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/ReadVariableOp2´
Xtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOpXtransformer_model_3/transformer_encoder/layer_normalization/batchnorm/mul/ReadVariableOp2°
Vtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOpVtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/ReadVariableOp2¸
Ztransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOpZtransformer_model_3/transformer_encoder/layer_normalization_1/batchnorm/mul/ReadVariableOp2¢
Otransformer_model_3/transformer_encoder/multi_head_attention/add/ReadVariableOpOtransformer_model_3/transformer_encoder/multi_head_attention/add/ReadVariableOp2¶
Ytransformer_model_3/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOpYtransformer_model_3/transformer_encoder/multi_head_attention/einsum/Einsum/ReadVariableOp2º
[transformer_model_3/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp[transformer_model_3/transformer_encoder/multi_head_attention/einsum_1/Einsum/ReadVariableOp2º
[transformer_model_3/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp[transformer_model_3/transformer_encoder/multi_head_attention/einsum_2/Einsum/ReadVariableOp2º
[transformer_model_3/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp[transformer_model_3/transformer_encoder/multi_head_attention/einsum_5/Einsum/ReadVariableOp:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:&"
 
_output_shapes
:


p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_343835

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
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ¢
´.
__inference__traced_save_345738
file_prefixV
Rsavev2_transformer_model_3_positional_embedding_dense_2_kernel_read_readvariableopT
Psavev2_transformer_model_3_positional_embedding_dense_2_bias_read_readvariableoph
dsavev2_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopf
bsavev2_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_read_readvariableoph
dsavev2_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopm
isavev2_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_read_readvariableopk
gsavev2_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_read_readvariableop`
\savev2_transformer_model_3_transformer_encoder_layer_normalization_gamma_read_readvariableop_
[savev2_transformer_model_3_transformer_encoder_layer_normalization_beta_read_readvariableopT
Psavev2_transformer_model_3_transformer_encoder_conv1d_kernel_read_readvariableopR
Nsavev2_transformer_model_3_transformer_encoder_conv1d_bias_read_readvariableopV
Rsavev2_transformer_model_3_transformer_encoder_conv1d_1_kernel_read_readvariableopT
Psavev2_transformer_model_3_transformer_encoder_conv1d_1_bias_read_readvariableopb
^savev2_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_read_readvariableopa
]savev2_transformer_model_3_transformer_encoder_layer_normalization_1_beta_read_readvariableop?
;savev2_transformer_model_3_dense_kernel_read_readvariableop=
9savev2_transformer_model_3_dense_bias_read_readvariableopA
=savev2_transformer_model_3_dense_1_kernel_read_readvariableop?
;savev2_transformer_model_3_dense_1_bias_read_readvariableop%
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
 savev2_count_read_readvariableop]
Ysavev2_adam_transformer_model_3_positional_embedding_dense_2_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_model_3_positional_embedding_dense_2_bias_m_read_readvariableopo
ksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_m_read_readvariableopm
isavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_m_read_readvariableopo
ksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_m_read_readvariableopt
psavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_m_read_readvariableopr
nsavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_m_read_readvariableopg
csavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_m_read_readvariableopf
bsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_m_read_readvariableop[
Wsavev2_adam_transformer_model_3_transformer_encoder_conv1d_kernel_m_read_readvariableopY
Usavev2_adam_transformer_model_3_transformer_encoder_conv1d_bias_m_read_readvariableop]
Ysavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_m_read_readvariableopi
esavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_m_read_readvariableoph
dsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_m_read_readvariableopF
Bsavev2_adam_transformer_model_3_dense_kernel_m_read_readvariableopD
@savev2_adam_transformer_model_3_dense_bias_m_read_readvariableopH
Dsavev2_adam_transformer_model_3_dense_1_kernel_m_read_readvariableopF
Bsavev2_adam_transformer_model_3_dense_1_bias_m_read_readvariableop]
Ysavev2_adam_transformer_model_3_positional_embedding_dense_2_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_model_3_positional_embedding_dense_2_bias_v_read_readvariableopo
ksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_v_read_readvariableopm
isavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_v_read_readvariableopo
ksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_v_read_readvariableopt
psavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_v_read_readvariableopr
nsavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_v_read_readvariableopg
csavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_v_read_readvariableopf
bsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_v_read_readvariableop[
Wsavev2_adam_transformer_model_3_transformer_encoder_conv1d_kernel_v_read_readvariableopY
Usavev2_adam_transformer_model_3_transformer_encoder_conv1d_bias_v_read_readvariableop]
Ysavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_v_read_readvariableopi
esavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_v_read_readvariableoph
dsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_v_read_readvariableopF
Bsavev2_adam_transformer_model_3_dense_kernel_v_read_readvariableopD
@savev2_adam_transformer_model_3_dense_bias_v_read_readvariableopH
Dsavev2_adam_transformer_model_3_dense_1_kernel_v_read_readvariableopF
Bsavev2_adam_transformer_model_3_dense_1_bias_v_read_readvariableop
savev2_const_1

identity_1¢MergeV2Checkpointsw
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
: Þ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
valueýBúEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHú
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*
valueBEB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¬-
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Rsavev2_transformer_model_3_positional_embedding_dense_2_kernel_read_readvariableopPsavev2_transformer_model_3_positional_embedding_dense_2_bias_read_readvariableopdsavev2_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_read_readvariableopbsavev2_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_read_readvariableopdsavev2_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_read_readvariableopisavev2_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_read_readvariableopgsavev2_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_read_readvariableop\savev2_transformer_model_3_transformer_encoder_layer_normalization_gamma_read_readvariableop[savev2_transformer_model_3_transformer_encoder_layer_normalization_beta_read_readvariableopPsavev2_transformer_model_3_transformer_encoder_conv1d_kernel_read_readvariableopNsavev2_transformer_model_3_transformer_encoder_conv1d_bias_read_readvariableopRsavev2_transformer_model_3_transformer_encoder_conv1d_1_kernel_read_readvariableopPsavev2_transformer_model_3_transformer_encoder_conv1d_1_bias_read_readvariableop^savev2_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_read_readvariableop]savev2_transformer_model_3_transformer_encoder_layer_normalization_1_beta_read_readvariableop;savev2_transformer_model_3_dense_kernel_read_readvariableop9savev2_transformer_model_3_dense_bias_read_readvariableop=savev2_transformer_model_3_dense_1_kernel_read_readvariableop;savev2_transformer_model_3_dense_1_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopYsavev2_adam_transformer_model_3_positional_embedding_dense_2_kernel_m_read_readvariableopWsavev2_adam_transformer_model_3_positional_embedding_dense_2_bias_m_read_readvariableopksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_m_read_readvariableopisavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_m_read_readvariableopksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_m_read_readvariableoppsavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_m_read_readvariableopnsavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_m_read_readvariableopcsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_m_read_readvariableopbsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_m_read_readvariableopWsavev2_adam_transformer_model_3_transformer_encoder_conv1d_kernel_m_read_readvariableopUsavev2_adam_transformer_model_3_transformer_encoder_conv1d_bias_m_read_readvariableopYsavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_m_read_readvariableopWsavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_m_read_readvariableopesavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_m_read_readvariableopdsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_m_read_readvariableopBsavev2_adam_transformer_model_3_dense_kernel_m_read_readvariableop@savev2_adam_transformer_model_3_dense_bias_m_read_readvariableopDsavev2_adam_transformer_model_3_dense_1_kernel_m_read_readvariableopBsavev2_adam_transformer_model_3_dense_1_bias_m_read_readvariableopYsavev2_adam_transformer_model_3_positional_embedding_dense_2_kernel_v_read_readvariableopWsavev2_adam_transformer_model_3_positional_embedding_dense_2_bias_v_read_readvariableopksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_query_kernel_v_read_readvariableopisavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_key_kernel_v_read_readvariableopksavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_value_kernel_v_read_readvariableoppsavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_kernel_v_read_readvariableopnsavev2_adam_transformer_model_3_transformer_encoder_multi_head_attention_projection_bias_v_read_readvariableopcsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_gamma_v_read_readvariableopbsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_beta_v_read_readvariableopWsavev2_adam_transformer_model_3_transformer_encoder_conv1d_kernel_v_read_readvariableopUsavev2_adam_transformer_model_3_transformer_encoder_conv1d_bias_v_read_readvariableopYsavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_kernel_v_read_readvariableopWsavev2_adam_transformer_model_3_transformer_encoder_conv1d_1_bias_v_read_readvariableopesavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_gamma_v_read_readvariableopdsavev2_adam_transformer_model_3_transformer_encoder_layer_normalization_1_beta_v_read_readvariableopBsavev2_adam_transformer_model_3_dense_kernel_v_read_readvariableop@savev2_adam_transformer_model_3_dense_bias_v_read_readvariableopDsavev2_adam_transformer_model_3_dense_1_kernel_v_read_readvariableopBsavev2_adam_transformer_model_3_dense_1_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
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
_input_shapesò
ï: :	:::::::::::::::
::	:: : : : : : : : : : : :	:::::::::::::::
::	::	:::::::::::::::
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
::*&
$
_output_shapes
::*&
$
_output_shapes
::*&
$
_output_shapes
::!
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
::*"&
$
_output_shapes
::*#&
$
_output_shapes
::*$&
$
_output_shapes
::!%
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
::*5&
$
_output_shapes
::*6&
$
_output_shapes
::*7&
$
_output_shapes
::!8
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
ù	
b
C__inference_dropout_layer_call_and_return_conditional_losses_344140

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

(__inference_dense_1_layer_call_fn_345191

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_344060o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
´
4__inference_transformer_model_3_layer_call_fn_344110
input_1
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
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
identity¢StatefulPartitionedCallÚ
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
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344067o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:&"
 
_output_shapes
:

Ú
a
C__inference_dropout_layer_call_and_return_conditional_losses_344047

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

õ
A__inference_dense_layer_call_and_return_conditional_losses_344036

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
õ
4__inference_transformer_encoder_layer_call_fn_345264

inputs
unknown:!
	unknown_0:!
	unknown_1:!
	unknown_2:
	unknown_3:	
	unknown_4:	
	unknown_5:	!
	unknown_6:
	unknown_7:	!
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	
identity¢StatefulPartitionedCallý
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
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_344297t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4
±
P__inference_positional_embedding_layer_call_and_return_conditional_losses_343899
x<
)dense_2_tensordot_readvariableop_resource:	6
'dense_2_biasadd_readvariableop_resource:	
unknown
identity¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp6
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
valueB:Ñ
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
value	B : Û
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
value	B : ß
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
value	B : ¼
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
:ÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
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
:ÿÿÿÿÿÿÿÿÿG
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
value	B : ­
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
value	B :§
strided_slice_1/stack_2Pack"strided_slice_1/stack_2/0:output:0Const_1:output:0"strided_slice_1/stack_2/2:output:0*
N*
T0*
_output_shapes
:þ
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
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityadd:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : :
2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

&
ê
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344067
x.
positional_embedding_343900:	*
positional_embedding_343902:	
positional_embedding_3439042
transformer_encoder_343997:2
transformer_encoder_343999:2
transformer_encoder_344001:2
transformer_encoder_344003:)
transformer_encoder_344005:	)
transformer_encoder_344007:	)
transformer_encoder_344009:	2
transformer_encoder_344011:)
transformer_encoder_344013:	2
transformer_encoder_344015:)
transformer_encoder_344017:	)
transformer_encoder_344019:	)
transformer_encoder_344021:	 
dense_344037:

dense_344039:	!
dense_1_344061:	
dense_1_344063:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢,positional_embedding/StatefulPartitionedCall¢+transformer_encoder/StatefulPartitionedCallÁ
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallxpositional_embedding_343900positional_embedding_343902positional_embedding_343904*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_343899
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_343997transformer_encoder_343999transformer_encoder_344001transformer_encoder_344003transformer_encoder_344005transformer_encoder_344007transformer_encoder_344009transformer_encoder_344011transformer_encoder_344013transformer_encoder_344015transformer_encoder_344017transformer_encoder_344019transformer_encoder_344021*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_343996
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_343835
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_344037dense_344039*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_344036Ú
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_344047
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_344061dense_1_344063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_344060w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

ö

O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345463

inputsR
:multi_head_attention_einsum_einsum_readvariableop_resource:T
<multi_head_attention_einsum_1_einsum_readvariableop_resource:T
<multi_head_attention_einsum_2_einsum_readvariableop_resource:T
<multi_head_attention_einsum_5_einsum_readvariableop_resource:?
0multi_head_attention_add_readvariableop_resource:	H
9layer_normalization_batchnorm_mul_readvariableop_resource:	D
5layer_normalization_batchnorm_readvariableop_resource:	J
2conv1d_conv1d_expanddims_1_readvariableop_resource:5
&conv1d_biasadd_readvariableop_resource:	L
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:7
(conv1d_1_biasadd_readvariableop_resource:	J
;layer_normalization_1_batchnorm_mul_readvariableop_resource:	F
7layer_normalization_1_batchnorm_readvariableop_resource:	
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢'multi_head_attention/add/ReadVariableOp¢1multi_head_attention/einsum/Einsum/ReadVariableOp¢3multi_head_attention/einsum_1/Einsum/ReadVariableOp¢3multi_head_attention/einsum_2/Einsum/ReadVariableOp¢3multi_head_attention/einsum_5/Einsum/ReadVariableOp²
1multi_head_attention/einsum/Einsum/ReadVariableOpReadVariableOp:multi_head_attention_einsum_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ô
"multi_head_attention/einsum/EinsumEinsuminputs9multi_head_attention/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NI,HIO->...NHO¶
3multi_head_attention/einsum_1/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_1_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_1/EinsumEinsuminputs;multi_head_attention/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...MI,HIO->...MHO¶
3multi_head_attention/einsum_2/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_2_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0Ø
$multi_head_attention/einsum_2/EinsumEinsuminputs;multi_head_attention/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
: ®
multi_head_attention/truedivRealDiv+multi_head_attention/einsum/Einsum:output:0multi_head_attention/Sqrt:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
$multi_head_attention/einsum_3/EinsumEinsum multi_head_attention/truediv:z:0-multi_head_attention/einsum_1/Einsum:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...NHO,...MHO->...HNM
multi_head_attention/SoftmaxSoftmax-multi_head_attention/einsum_3/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,multi_head_attention/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?Ê
*multi_head_attention/dropout_1/dropout/MulMul&multi_head_attention/Softmax:softmax:05multi_head_attention/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,multi_head_attention/dropout_1/dropout/ShapeShape&multi_head_attention/Softmax:softmax:0*
T0*
_output_shapes
:Ò
Cmulti_head_attention/dropout_1/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿµ
+multi_head_attention/dropout_1/dropout/CastCast7multi_head_attention/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
,multi_head_attention/dropout_1/dropout/Mul_1Mul.multi_head_attention/dropout_1/dropout/Mul:z:0/multi_head_attention/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
$multi_head_attention/einsum_4/EinsumEinsum0multi_head_attention/dropout_1/dropout/Mul_1:z:0-multi_head_attention/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
equation...HNM,...MHI->...NHI¶
3multi_head_attention/einsum_5/Einsum/ReadVariableOpReadVariableOp<multi_head_attention_einsum_5_einsum_readvariableop_resource*$
_output_shapes
:*
dtype0û
$multi_head_attention/einsum_5/EinsumEinsum-multi_head_attention/einsum_4/Einsum:output:0;multi_head_attention/einsum_5/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
equation...NHI,HIO->...NO
'multi_head_attention/add/ReadVariableOpReadVariableOp0multi_head_attention_add_readvariableop_resource*
_output_shapes	
:*
dtype0¸
multi_head_attention/addAddV2-multi_head_attention/einsum_5/Einsum:output:0/multi_head_attention/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_2/dropout/MulMulmulti_head_attention/add:z:0 dropout_2/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dropout_2/dropout/ShapeShapemulti_head_attention/add:z:0*
T0*
_output_shapes
:¥
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>É
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:É
 layer_normalization/moments/meanMeandropout_2/dropout/Mul_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout_2/dropout/Mul_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ç
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75½
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Â
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
#layer_normalization/batchnorm/mul_1Muldropout_2/dropout/Mul_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¾
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
addAddV2'layer_normalization/batchnorm/add_1:z:0inputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/Conv1D/ExpandDims
ExpandDimsadd:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ·
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:Ã
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_3/dropout/MulMulconv1d/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dropout_3/dropout/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:¥
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>É
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ©
conv1d_1/Conv1D/ExpandDims
ExpandDimsdropout_3/dropout/Mul_1:z:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ½
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:É
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ë
"layer_normalization_1/moments/meanMeanconv1d_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0È
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
%layer_normalization_1/batchnorm/mul_1Mulconv1d_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
add_1AddV2add:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
IdentityIdentity	add_1:z:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp(^multi_head_attention/add/ReadVariableOp2^multi_head_attention/einsum/Einsum/ReadVariableOp4^multi_head_attention/einsum_1/Einsum/ReadVariableOp4^multi_head_attention/einsum_2/Einsum/ReadVariableOp4^multi_head_attention/einsum_5/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2>
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
®
5__inference_positional_embedding_layer_call_fn_345117
x
unknown:	
	unknown_0:	
	unknown_1
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_343899t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : :
22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

®
®
4__inference_transformer_model_3_layer_call_fn_344766
x
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
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
identity¢StatefulPartitionedCallÔ
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
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344435o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

Ú
a
C__inference_dropout_layer_call_and_return_conditional_losses_345498

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
´
4__inference_transformer_model_3_layer_call_fn_344523
input_1
unknown:	
	unknown_0:	
	unknown_1!
	unknown_2:!
	unknown_3:!
	unknown_4:!
	unknown_5:
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
identity¢StatefulPartitionedCallÚ
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
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344435o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:&"
 
_output_shapes
:

»'
	
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344435
x.
positional_embedding_344388:	*
positional_embedding_344390:	
positional_embedding_3443922
transformer_encoder_344395:2
transformer_encoder_344397:2
transformer_encoder_344399:2
transformer_encoder_344401:)
transformer_encoder_344403:	)
transformer_encoder_344405:	)
transformer_encoder_344407:	2
transformer_encoder_344409:)
transformer_encoder_344411:	2
transformer_encoder_344413:)
transformer_encoder_344415:	)
transformer_encoder_344417:	)
transformer_encoder_344419:	 
dense_344423:

dense_344425:	!
dense_1_344429:	
dense_1_344431:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢,positional_embedding/StatefulPartitionedCall¢+transformer_encoder/StatefulPartitionedCallÁ
,positional_embedding/StatefulPartitionedCallStatefulPartitionedCallxpositional_embedding_344388positional_embedding_344390positional_embedding_344392*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_positional_embedding_layer_call_and_return_conditional_losses_343899
+transformer_encoder/StatefulPartitionedCallStatefulPartitionedCall5positional_embedding/StatefulPartitionedCall:output:0transformer_encoder_344395transformer_encoder_344397transformer_encoder_344399transformer_encoder_344401transformer_encoder_344403transformer_encoder_344405transformer_encoder_344407transformer_encoder_344409transformer_encoder_344411transformer_encoder_344413transformer_encoder_344415transformer_encoder_344417transformer_encoder_344419*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_344297
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_343835
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_344423dense_344425*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_344036ê
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_344140
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_344429dense_1_344431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_344060w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall-^positional_embedding/StatefulPartitionedCall,^transformer_encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:ÿÿÿÿÿÿÿÿÿ: : :
: : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2\
,positional_embedding/StatefulPartitionedCall,positional_embedding/StatefulPartitionedCall2Z
+transformer_encoder/StatefulPartitionedCall+transformer_encoder/StatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex:&"
 
_output_shapes
:

þ
U
9__inference_global_average_pooling1d_layer_call_fn_345176

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_343835i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¯
serving_default
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:´³
Æ
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
®
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
®
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
Ê
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
ô
(trace_0
)trace_1
*trace_2
+trace_32
4__inference_transformer_model_3_layer_call_fn_344110
4__inference_transformer_model_3_layer_call_fn_344721
4__inference_transformer_model_3_layer_call_fn_344766
4__inference_transformer_model_3_layer_call_fn_344523®
¥²¡
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
annotationsª *
 z(trace_0z)trace_1z*trace_2z+trace_3
à
,trace_0
-trace_1
.trace_2
/trace_32õ
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344922
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_345106
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344573
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344623®
¥²¡
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
annotationsª *
 z,trace_0z-trace_1z.trace_2z/trace_3
ÌBÉ
!__inference__wrapped_model_343825input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
Ï

0beta_1

1beta_2
	2decay
3learning_rate
4itermmmmmmmmmmmmmmmm  m¡!m¢"m£v¤v¥v¦v§v¨v©vªv«v¬v­v®v¯v°v±v²v³ v´!vµ"v¶"
	optimizer
´
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
¥
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
»
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
J:H	27transformer_model_3/positional_embedding/dense_2/kernel
D:B25transformer_model_3/positional_embedding/dense_2/bias
a:_2Itransformer_model_3/transformer_encoder/multi_head_attention/query_kernel
_:]2Gtransformer_model_3/transformer_encoder/multi_head_attention/key_kernel
a:_2Itransformer_model_3/transformer_encoder/multi_head_attention/value_kernel
f:d2Ntransformer_model_3/transformer_encoder/multi_head_attention/projection_kernel
[:Y2Ltransformer_model_3/transformer_encoder/multi_head_attention/projection_bias
P:N2Atransformer_model_3/transformer_encoder/layer_normalization/gamma
O:M2@transformer_model_3/transformer_encoder/layer_normalization/beta
M:K25transformer_model_3/transformer_encoder/conv1d/kernel
B:@23transformer_model_3/transformer_encoder/conv1d/bias
O:M27transformer_model_3/transformer_encoder/conv1d_1/kernel
D:B25transformer_model_3/transformer_encoder/conv1d_1/bias
R:P2Ctransformer_model_3/transformer_encoder/layer_normalization_1/gamma
Q:O2Btransformer_model_3/transformer_encoder/layer_normalization_1/beta
4:2
2 transformer_model_3/dense/kernel
-:+2transformer_model_3/dense/bias
5:3	2"transformer_model_3/dense_1/kernel
.:,2 transformer_model_3/dense_1/bias
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
õBò
4__inference_transformer_model_3_layer_call_fn_344110input_1"®
¥²¡
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
annotationsª *
 
ïBì
4__inference_transformer_model_3_layer_call_fn_344721x"®
¥²¡
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
annotationsª *
 
ïBì
4__inference_transformer_model_3_layer_call_fn_344766x"®
¥²¡
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
annotationsª *
 
õBò
4__inference_transformer_model_3_layer_call_fn_344523input_1"®
¥²¡
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
annotationsª *
 
B
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344922x"®
¥²¡
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
annotationsª *
 
B
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_345106x"®
¥²¡
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
annotationsª *
 
B
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344573input_1"®
¥²¡
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
annotationsª *
 
B
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344623input_1"®
¥²¡
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
annotationsª *
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
­
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
ô
Ttrace_02×
5__inference_positional_embedding_layer_call_fn_345117
²
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
annotationsª *
 zTtrace_0

Utrace_02ò
P__inference_positional_embedding_layer_call_and_return_conditional_losses_345171
²
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
annotationsª *
 zUtrace_0
»
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
­
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
otrace_02í
9__inference_global_average_pooling1d_layer_call_fn_345176¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0
¥
ptrace_02
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_345182¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zptrace_0
»
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
¼
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
°
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
î
trace_02Ï
(__inference_dense_1_layer_call_fn_345191¢
²
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
annotationsª *
 ztrace_0

trace_02ê
C__inference_dense_1_layer_call_and_return_conditional_losses_345202¢
²
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
annotationsª *
 ztrace_0
ËBÈ
$__inference_signature_wrapper_344676input_1"
²
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
annotationsª *
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
ßBÜ
5__inference_positional_embedding_layer_call_fn_345117x"
²
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
annotationsª *
 
úB÷
P__inference_positional_embedding_layer_call_and_return_conditional_losses_345171x"
²
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
annotationsª *
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
²
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
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
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
²
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
Ý
trace_0
trace_12¢
4__inference_transformer_encoder_layer_call_fn_345233
4__inference_transformer_encoder_layer_call_fn_345264³
ª²¦
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
annotationsª *
 ztrace_0ztrace_1

trace_0
 trace_12Ø
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345353
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345463³
ª²¦
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
annotationsª *
 ztrace_0z trace_1
 "
trackable_list_wrapper

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§dropout
query_kernel

key_kernel
value_kernel
projection_kernel
projection_bias"
_tf_keras_layer
Ã
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses
®_random_generator"
_tf_keras_layer
Ë
¯	variables
°trainable_variables
±regularization_losses
²	keras_api
³__call__
+´&call_and_return_all_conditional_losses
	µaxis
	gamma
beta"
_tf_keras_layer
ä
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
º__call__
+»&call_and_return_all_conditional_losses

kernel
bias
!¼_jit_compiled_convolution_op"
_tf_keras_layer
Ã
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses
Ã_random_generator"
_tf_keras_layer
ä
Ä	variables
Åtrainable_variables
Æregularization_losses
Ç	keras_api
È__call__
+É&call_and_return_all_conditional_losses

kernel
bias
!Ê_jit_compiled_convolution_op"
_tf_keras_layer
Ë
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses
	Ñaxis
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
úB÷
9__inference_global_average_pooling1d_layer_call_fn_345176inputs"¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_345182inputs"¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ì
×trace_02Í
&__inference_dense_layer_call_fn_345472¢
²
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
annotationsª *
 z×trace_0

Øtrace_02è
A__inference_dense_layer_call_and_return_conditional_losses_345483¢
²
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
annotationsª *
 zØtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
Æ
Þtrace_0
ßtrace_12
(__inference_dropout_layer_call_fn_345488
(__inference_dropout_layer_call_fn_345493´
«²§
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
kwonlydefaultsª 
annotationsª *
 zÞtrace_0zßtrace_1
ü
àtrace_0
átrace_12Á
C__inference_dropout_layer_call_and_return_conditional_losses_345498
C__inference_dropout_layer_call_and_return_conditional_losses_345510´
«²§
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
kwonlydefaultsª 
annotationsª *
 zàtrace_0zátrace_1
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
ÜBÙ
(__inference_dense_1_layer_call_fn_345191inputs"¢
²
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
annotationsª *
 
÷Bô
C__inference_dense_1_layer_call_and_return_conditional_losses_345202inputs"¢
²
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
annotationsª *
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
ùBö
4__inference_transformer_encoder_layer_call_fn_345233inputs"³
ª²¦
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
annotationsª *
 
ùBö
4__inference_transformer_encoder_layer_call_fn_345264inputs"³
ª²¦
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
annotationsª *
 
B
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345353inputs"³
ª²¦
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
annotationsª *
 
B
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345463inputs"³
ª²¦
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
annotationsª *
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
¸
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
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
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
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
kwonlydefaultsª 
annotationsª *
 
Ã
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses
í_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
¨	variables
©trainable_variables
ªregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
º2·´
«²§
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
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
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
kwonlydefaultsª 
annotationsª *
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
¸
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
¯	variables
°trainable_variables
±regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
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
¸
ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
¶	variables
·trainable_variables
¸regularization_losses
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
´2±®
£²
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
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
º2·´
«²§
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
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
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
kwonlydefaultsª 
annotationsª *
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ä	variables
Åtrainable_variables
Æregularization_losses
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
´2±®
£²
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
annotationsª *
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
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
ÚB×
&__inference_dense_layer_call_fn_345472inputs"¢
²
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
annotationsª *
 
õBò
A__inference_dense_layer_call_and_return_conditional_losses_345483inputs"¢
²
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
annotationsª *
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
îBë
(__inference_dropout_layer_call_fn_345488inputs"´
«²§
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
kwonlydefaultsª 
annotationsª *
 
îBë
(__inference_dropout_layer_call_fn_345493inputs"´
«²§
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
kwonlydefaultsª 
annotationsª *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_345498inputs"´
«²§
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
kwonlydefaultsª 
annotationsª *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_345510inputs"´
«²§
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
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
(
§0"
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
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
º2·´
«²§
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
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
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
kwonlydefaultsª 
annotationsª *
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
O:M	2>Adam/transformer_model_3/positional_embedding/dense_2/kernel/m
I:G2<Adam/transformer_model_3/positional_embedding/dense_2/bias/m
f:d2PAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/m
d:b2NAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/m
f:d2PAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/m
k:i2UAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/m
`:^2SAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/m
U:S2HAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/m
T:R2GAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/m
R:P2<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/m
G:E2:Adam/transformer_model_3/transformer_encoder/conv1d/bias/m
T:R2>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/m
I:G2<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/m
W:U2JAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/m
V:T2IAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/m
9:7
2'Adam/transformer_model_3/dense/kernel/m
2:02%Adam/transformer_model_3/dense/bias/m
::8	2)Adam/transformer_model_3/dense_1/kernel/m
3:12'Adam/transformer_model_3/dense_1/bias/m
O:M	2>Adam/transformer_model_3/positional_embedding/dense_2/kernel/v
I:G2<Adam/transformer_model_3/positional_embedding/dense_2/bias/v
f:d2PAdam/transformer_model_3/transformer_encoder/multi_head_attention/query_kernel/v
d:b2NAdam/transformer_model_3/transformer_encoder/multi_head_attention/key_kernel/v
f:d2PAdam/transformer_model_3/transformer_encoder/multi_head_attention/value_kernel/v
k:i2UAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_kernel/v
`:^2SAdam/transformer_model_3/transformer_encoder/multi_head_attention/projection_bias/v
U:S2HAdam/transformer_model_3/transformer_encoder/layer_normalization/gamma/v
T:R2GAdam/transformer_model_3/transformer_encoder/layer_normalization/beta/v
R:P2<Adam/transformer_model_3/transformer_encoder/conv1d/kernel/v
G:E2:Adam/transformer_model_3/transformer_encoder/conv1d/bias/v
T:R2>Adam/transformer_model_3/transformer_encoder/conv1d_1/kernel/v
I:G2<Adam/transformer_model_3/transformer_encoder/conv1d_1/bias/v
W:U2JAdam/transformer_model_3/transformer_encoder/layer_normalization_1/gamma/v
V:T2IAdam/transformer_model_3/transformer_encoder/layer_normalization_1/beta/v
9:7
2'Adam/transformer_model_3/dense/kernel/v
2:02%Adam/transformer_model_3/dense/bias/v
::8	2)Adam/transformer_model_3/dense_1/kernel/v
3:12'Adam/transformer_model_3/dense_1/bias/v
J
Constjtf.TrackableConstant¨
!__inference__wrapped_model_343825· !"4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_1_layer_call_and_return_conditional_losses_345202]!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_1_layer_call_fn_345191P!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_dense_layer_call_and_return_conditional_losses_345483^ 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dense_layer_call_fn_345472Q 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_dropout_layer_call_and_return_conditional_losses_345498^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¥
C__inference_dropout_layer_call_and_return_conditional_losses_345510^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dropout_layer_call_fn_345488Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
(__inference_dropout_layer_call_fn_345493Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÓ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_345182{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 «
9__inference_global_average_pooling1d_layer_call_fn_345176nI¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¶
P__inference_positional_embedding_layer_call_and_return_conditional_losses_345171b·.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
5__inference_positional_embedding_layer_call_fn_345117U·.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¶
$__inference_signature_wrapper_344676· !"?¢<
¢ 
5ª2
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿÈ
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345353u8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 È
O__inference_transformer_encoder_layer_call_and_return_conditional_losses_345463u8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
  
4__inference_transformer_encoder_layer_call_fn_345233h8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ 
4__inference_transformer_encoder_layer_call_fn_345264h8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿË
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344573x· !"8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344623x· !"8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_344922r· !"2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
O__inference_transformer_model_3_layer_call_and_return_conditional_losses_345106r· !"2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
4__inference_transformer_model_3_layer_call_fn_344110k· !"8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ£
4__inference_transformer_model_3_layer_call_fn_344523k· !"8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
4__inference_transformer_model_3_layer_call_fn_344721e· !"2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
4__inference_transformer_model_3_layer_call_fn_344766e· !"2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ