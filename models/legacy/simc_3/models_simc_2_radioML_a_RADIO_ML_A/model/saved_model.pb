Щ№
Ѕє
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
М
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
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

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58
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
v
SGD/m/FC1_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameSGD/m/FC1_/bias
o
#SGD/m/FC1_/bias/Read/ReadVariableOpReadVariableOpSGD/m/FC1_/bias*
_output_shapes
:*
dtype0
~
SGD/m/FC1_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*"
shared_nameSGD/m/FC1_/kernel
w
%SGD/m/FC1_/kernel/Read/ReadVariableOpReadVariableOpSGD/m/FC1_/kernel*
_output_shapes

:`*
dtype0

SGD/m/BN6_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*.
shared_nameSGD/m/BN6_/custom_batch_gamma

1SGD/m/BN6_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpSGD/m/BN6_/custom_batch_gamma*
_output_shapes
:`*
dtype0

SGD/m/BN6_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_nameSGD/m/BN6_/custom_batch_beta

0SGD/m/BN6_/custom_batch_beta/Read/ReadVariableOpReadVariableOpSGD/m/BN6_/custom_batch_beta*
_output_shapes
:`*
dtype0
x
SGD/m/CNN6_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*!
shared_nameSGD/m/CNN6_/bias
q
$SGD/m/CNN6_/bias/Read/ReadVariableOpReadVariableOpSGD/m/CNN6_/bias*
_output_shapes
:`*
dtype0

SGD/m/CNN6_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*#
shared_nameSGD/m/CNN6_/kernel

&SGD/m/CNN6_/kernel/Read/ReadVariableOpReadVariableOpSGD/m/CNN6_/kernel*&
_output_shapes
:@`*
dtype0

SGD/m/BN5_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameSGD/m/BN5_/custom_batch_gamma

1SGD/m/BN5_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpSGD/m/BN5_/custom_batch_gamma*
_output_shapes
:@*
dtype0

SGD/m/BN5_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameSGD/m/BN5_/custom_batch_beta

0SGD/m/BN5_/custom_batch_beta/Read/ReadVariableOpReadVariableOpSGD/m/BN5_/custom_batch_beta*
_output_shapes
:@*
dtype0
x
SGD/m/CNN5_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameSGD/m/CNN5_/bias
q
$SGD/m/CNN5_/bias/Read/ReadVariableOpReadVariableOpSGD/m/CNN5_/bias*
_output_shapes
:@*
dtype0

SGD/m/CNN5_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*#
shared_nameSGD/m/CNN5_/kernel

&SGD/m/CNN5_/kernel/Read/ReadVariableOpReadVariableOpSGD/m/CNN5_/kernel*&
_output_shapes
:0@*
dtype0

SGD/m/BN4_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nameSGD/m/BN4_/custom_batch_gamma

1SGD/m/BN4_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpSGD/m/BN4_/custom_batch_gamma*
_output_shapes
:0*
dtype0

SGD/m/BN4_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_nameSGD/m/BN4_/custom_batch_beta

0SGD/m/BN4_/custom_batch_beta/Read/ReadVariableOpReadVariableOpSGD/m/BN4_/custom_batch_beta*
_output_shapes
:0*
dtype0
x
SGD/m/CNN4_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_nameSGD/m/CNN4_/bias
q
$SGD/m/CNN4_/bias/Read/ReadVariableOpReadVariableOpSGD/m/CNN4_/bias*
_output_shapes
:0*
dtype0

SGD/m/CNN4_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*#
shared_nameSGD/m/CNN4_/kernel

&SGD/m/CNN4_/kernel/Read/ReadVariableOpReadVariableOpSGD/m/CNN4_/kernel*&
_output_shapes
: 0*
dtype0

SGD/m/BN3_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/m/BN3_/custom_batch_gamma

1SGD/m/BN3_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpSGD/m/BN3_/custom_batch_gamma*
_output_shapes
: *
dtype0

SGD/m/BN3_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameSGD/m/BN3_/custom_batch_beta

0SGD/m/BN3_/custom_batch_beta/Read/ReadVariableOpReadVariableOpSGD/m/BN3_/custom_batch_beta*
_output_shapes
: *
dtype0
x
SGD/m/CNN3_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameSGD/m/CNN3_/bias
q
$SGD/m/CNN3_/bias/Read/ReadVariableOpReadVariableOpSGD/m/CNN3_/bias*
_output_shapes
: *
dtype0

SGD/m/CNN3_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameSGD/m/CNN3_/kernel

&SGD/m/CNN3_/kernel/Read/ReadVariableOpReadVariableOpSGD/m/CNN3_/kernel*&
_output_shapes
: *
dtype0

SGD/m/BN2_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/m/BN2_/custom_batch_gamma

1SGD/m/BN2_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpSGD/m/BN2_/custom_batch_gamma*
_output_shapes
:*
dtype0

SGD/m/BN2_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameSGD/m/BN2_/custom_batch_beta

0SGD/m/BN2_/custom_batch_beta/Read/ReadVariableOpReadVariableOpSGD/m/BN2_/custom_batch_beta*
_output_shapes
:*
dtype0
x
SGD/m/CNN2_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameSGD/m/CNN2_/bias
q
$SGD/m/CNN2_/bias/Read/ReadVariableOpReadVariableOpSGD/m/CNN2_/bias*
_output_shapes
:*
dtype0

SGD/m/CNN2_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameSGD/m/CNN2_/kernel

&SGD/m/CNN2_/kernel/Read/ReadVariableOpReadVariableOpSGD/m/CNN2_/kernel*&
_output_shapes
:*
dtype0

SGD/m/BN1_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameSGD/m/BN1_/custom_batch_gamma

1SGD/m/BN1_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpSGD/m/BN1_/custom_batch_gamma*
_output_shapes
:*
dtype0

SGD/m/BN1_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameSGD/m/BN1_/custom_batch_beta

0SGD/m/BN1_/custom_batch_beta/Read/ReadVariableOpReadVariableOpSGD/m/BN1_/custom_batch_beta*
_output_shapes
:*
dtype0
x
SGD/m/CNN1_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameSGD/m/CNN1_/bias
q
$SGD/m/CNN1_/bias/Read/ReadVariableOpReadVariableOpSGD/m/CNN1_/bias*
_output_shapes
:*
dtype0

SGD/m/CNN1_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameSGD/m/CNN1_/kernel

&SGD/m/CNN1_/kernel/Read/ReadVariableOpReadVariableOpSGD/m/CNN1_/kernel*&
_output_shapes
:*
dtype0
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
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
j
	FC1_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	FC1_/bias
c
FC1_/bias/Read/ReadVariableOpReadVariableOp	FC1_/bias*
_output_shapes
:*
dtype0
r
FC1_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*
shared_nameFC1_/kernel
k
FC1_/kernel/Read/ReadVariableOpReadVariableOpFC1_/kernel*
_output_shapes

:`*
dtype0

!BN6_/custom_batch_moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!BN6_/custom_batch_moving_variance

5BN6_/custom_batch_moving_variance/Read/ReadVariableOpReadVariableOp!BN6_/custom_batch_moving_variance*
_output_shapes
:`*
dtype0

BN6_/custom_batch_moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*.
shared_nameBN6_/custom_batch_moving_mean

1BN6_/custom_batch_moving_mean/Read/ReadVariableOpReadVariableOpBN6_/custom_batch_moving_mean*
_output_shapes
:`*
dtype0

BN6_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*(
shared_nameBN6_/custom_batch_gamma

+BN6_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpBN6_/custom_batch_gamma*
_output_shapes
:`*
dtype0

BN6_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*'
shared_nameBN6_/custom_batch_beta
}
*BN6_/custom_batch_beta/Read/ReadVariableOpReadVariableOpBN6_/custom_batch_beta*
_output_shapes
:`*
dtype0
l

CNN6_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_name
CNN6_/bias
e
CNN6_/bias/Read/ReadVariableOpReadVariableOp
CNN6_/bias*
_output_shapes
:`*
dtype0
|
CNN6_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*
shared_nameCNN6_/kernel
u
 CNN6_/kernel/Read/ReadVariableOpReadVariableOpCNN6_/kernel*&
_output_shapes
:@`*
dtype0

!BN5_/custom_batch_moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!BN5_/custom_batch_moving_variance

5BN5_/custom_batch_moving_variance/Read/ReadVariableOpReadVariableOp!BN5_/custom_batch_moving_variance*
_output_shapes
:@*
dtype0

BN5_/custom_batch_moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameBN5_/custom_batch_moving_mean

1BN5_/custom_batch_moving_mean/Read/ReadVariableOpReadVariableOpBN5_/custom_batch_moving_mean*
_output_shapes
:@*
dtype0

BN5_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameBN5_/custom_batch_gamma

+BN5_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpBN5_/custom_batch_gamma*
_output_shapes
:@*
dtype0

BN5_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameBN5_/custom_batch_beta
}
*BN5_/custom_batch_beta/Read/ReadVariableOpReadVariableOpBN5_/custom_batch_beta*
_output_shapes
:@*
dtype0
l

CNN5_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
CNN5_/bias
e
CNN5_/bias/Read/ReadVariableOpReadVariableOp
CNN5_/bias*
_output_shapes
:@*
dtype0
|
CNN5_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*
shared_nameCNN5_/kernel
u
 CNN5_/kernel/Read/ReadVariableOpReadVariableOpCNN5_/kernel*&
_output_shapes
:0@*
dtype0

!BN4_/custom_batch_moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!BN4_/custom_batch_moving_variance

5BN4_/custom_batch_moving_variance/Read/ReadVariableOpReadVariableOp!BN4_/custom_batch_moving_variance*
_output_shapes
:0*
dtype0

BN4_/custom_batch_moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_nameBN4_/custom_batch_moving_mean

1BN4_/custom_batch_moving_mean/Read/ReadVariableOpReadVariableOpBN4_/custom_batch_moving_mean*
_output_shapes
:0*
dtype0

BN4_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameBN4_/custom_batch_gamma

+BN4_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpBN4_/custom_batch_gamma*
_output_shapes
:0*
dtype0

BN4_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameBN4_/custom_batch_beta
}
*BN4_/custom_batch_beta/Read/ReadVariableOpReadVariableOpBN4_/custom_batch_beta*
_output_shapes
:0*
dtype0
l

CNN4_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name
CNN4_/bias
e
CNN4_/bias/Read/ReadVariableOpReadVariableOp
CNN4_/bias*
_output_shapes
:0*
dtype0
|
CNN4_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*
shared_nameCNN4_/kernel
u
 CNN4_/kernel/Read/ReadVariableOpReadVariableOpCNN4_/kernel*&
_output_shapes
: 0*
dtype0

!BN3_/custom_batch_moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!BN3_/custom_batch_moving_variance

5BN3_/custom_batch_moving_variance/Read/ReadVariableOpReadVariableOp!BN3_/custom_batch_moving_variance*
_output_shapes
: *
dtype0

BN3_/custom_batch_moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameBN3_/custom_batch_moving_mean

1BN3_/custom_batch_moving_mean/Read/ReadVariableOpReadVariableOpBN3_/custom_batch_moving_mean*
_output_shapes
: *
dtype0

BN3_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameBN3_/custom_batch_gamma

+BN3_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpBN3_/custom_batch_gamma*
_output_shapes
: *
dtype0

BN3_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameBN3_/custom_batch_beta
}
*BN3_/custom_batch_beta/Read/ReadVariableOpReadVariableOpBN3_/custom_batch_beta*
_output_shapes
: *
dtype0
l

CNN3_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
CNN3_/bias
e
CNN3_/bias/Read/ReadVariableOpReadVariableOp
CNN3_/bias*
_output_shapes
: *
dtype0
|
CNN3_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameCNN3_/kernel
u
 CNN3_/kernel/Read/ReadVariableOpReadVariableOpCNN3_/kernel*&
_output_shapes
: *
dtype0

!BN2_/custom_batch_moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!BN2_/custom_batch_moving_variance

5BN2_/custom_batch_moving_variance/Read/ReadVariableOpReadVariableOp!BN2_/custom_batch_moving_variance*
_output_shapes
:*
dtype0

BN2_/custom_batch_moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBN2_/custom_batch_moving_mean

1BN2_/custom_batch_moving_mean/Read/ReadVariableOpReadVariableOpBN2_/custom_batch_moving_mean*
_output_shapes
:*
dtype0

BN2_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameBN2_/custom_batch_gamma

+BN2_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpBN2_/custom_batch_gamma*
_output_shapes
:*
dtype0

BN2_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBN2_/custom_batch_beta
}
*BN2_/custom_batch_beta/Read/ReadVariableOpReadVariableOpBN2_/custom_batch_beta*
_output_shapes
:*
dtype0
l

CNN2_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
CNN2_/bias
e
CNN2_/bias/Read/ReadVariableOpReadVariableOp
CNN2_/bias*
_output_shapes
:*
dtype0
|
CNN2_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCNN2_/kernel
u
 CNN2_/kernel/Read/ReadVariableOpReadVariableOpCNN2_/kernel*&
_output_shapes
:*
dtype0

!BN1_/custom_batch_moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!BN1_/custom_batch_moving_variance

5BN1_/custom_batch_moving_variance/Read/ReadVariableOpReadVariableOp!BN1_/custom_batch_moving_variance*
_output_shapes
:*
dtype0

BN1_/custom_batch_moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBN1_/custom_batch_moving_mean

1BN1_/custom_batch_moving_mean/Read/ReadVariableOpReadVariableOpBN1_/custom_batch_moving_mean*
_output_shapes
:*
dtype0

BN1_/custom_batch_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameBN1_/custom_batch_gamma

+BN1_/custom_batch_gamma/Read/ReadVariableOpReadVariableOpBN1_/custom_batch_gamma*
_output_shapes
:*
dtype0

BN1_/custom_batch_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameBN1_/custom_batch_beta
}
*BN1_/custom_batch_beta/Read/ReadVariableOpReadVariableOpBN1_/custom_batch_beta*
_output_shapes
:*
dtype0
l

CNN1_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
CNN1_/bias
e
CNN1_/bias/Read/ReadVariableOpReadVariableOp
CNN1_/bias*
_output_shapes
:*
dtype0
|
CNN1_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCNN1_/kernel
u
 CNN1_/kernel/Read/ReadVariableOpReadVariableOpCNN1_/kernel*&
_output_shapes
:*
dtype0

serving_default_input_1Placeholder*0
_output_shapes
:џџџџџџџџџ*
dtype0*%
shape:џџџџџџџџџ
ц	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1CNN1_/kernel
CNN1_/biasBN1_/custom_batch_moving_mean!BN1_/custom_batch_moving_varianceBN1_/custom_batch_gammaBN1_/custom_batch_betaCNN2_/kernel
CNN2_/biasBN2_/custom_batch_moving_mean!BN2_/custom_batch_moving_varianceBN2_/custom_batch_gammaBN2_/custom_batch_betaCNN3_/kernel
CNN3_/biasBN3_/custom_batch_moving_mean!BN3_/custom_batch_moving_varianceBN3_/custom_batch_gammaBN3_/custom_batch_betaCNN4_/kernel
CNN4_/biasBN4_/custom_batch_moving_mean!BN4_/custom_batch_moving_varianceBN4_/custom_batch_gammaBN4_/custom_batch_betaCNN5_/kernel
CNN5_/biasBN5_/custom_batch_moving_mean!BN5_/custom_batch_moving_varianceBN5_/custom_batch_gammaBN5_/custom_batch_betaCNN6_/kernel
CNN6_/biasBN6_/custom_batch_moving_mean!BN6_/custom_batch_moving_varianceBN6_/custom_batch_gammaBN6_/custom_batch_betaFC1_/kernel	FC1_/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_16093

NoOpNoOp
іи
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Аи
valueЅиBЁи Bи
­
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer-25
layer_with_weights-12
layer-26
layer-27
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&
signatures*
* 
Ш
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op*
К
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6custom_batch_beta
6beta
7custom_batch_gamma
	7gamma
8custom_batch_moving_mean
8moving_mean
 9custom_batch_moving_variance
9moving_variance*

:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 

@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
Ш
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op*
К
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Ucustom_batch_beta
Ubeta
Vcustom_batch_gamma
	Vgamma
Wcustom_batch_moving_mean
Wmoving_mean
 Xcustom_batch_moving_variance
Xmoving_variance*

Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 

_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
Ш
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op*
К
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
tcustom_batch_beta
tbeta
ucustom_batch_gamma
	ugamma
vcustom_batch_moving_mean
vmoving_mean
 wcustom_batch_moving_variance
wmoving_variance*

x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses* 

~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
custom_batch_beta
	beta
custom_batch_gamma

gamma
custom_batch_moving_mean
moving_mean
!custom_batch_moving_variance
moving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses* 
б
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses
Љkernel
	Њbias
!Ћ_jit_compiled_convolution_op*
Ш
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вcustom_batch_beta
	Вbeta
Гcustom_batch_gamma

Гgamma
Дcustom_batch_moving_mean
Дmoving_mean
!Еcustom_batch_moving_variance
Еmoving_variance*

Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses* 

М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses* 
б
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Шkernel
	Щbias
!Ъ_jit_compiled_convolution_op*
Ш
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
бcustom_batch_beta
	бbeta
вcustom_batch_gamma

вgamma
гcustom_batch_moving_mean
гmoving_mean
!дcustom_batch_moving_variance
дmoving_variance*

е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses* 

л	variables
мtrainable_variables
нregularization_losses
о	keras_api
п__call__
+р&call_and_return_all_conditional_losses* 

с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses* 
Ў
ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses
эkernel
	юbias*

я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses* 

ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses* 
О
-0
.1
62
73
84
95
L6
M7
U8
V9
W10
X11
k12
l13
t14
u15
v16
w17
18
19
20
21
22
23
Љ24
Њ25
В26
Г27
Д28
Е29
Ш30
Щ31
б32
в33
г34
д35
э36
ю37*
и
-0
.1
62
73
L4
M5
U6
V7
k8
l9
t10
u11
12
13
14
15
Љ16
Њ17
В18
Г19
Ш20
Щ21
б22
в23
э24
ю25*
* 
Е
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
u

_variables
_iterations
_learning_rate
_index_dict
	momentums
_update_step_xla*

serving_default* 

-0
.1*

-0
.1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

trace_0* 

trace_0* 
\V
VARIABLE_VALUECNN1_/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN1_/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
60
71
82
93*

60
71*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
qk
VARIABLE_VALUEBN1_/custom_batch_betaAlayer_with_weights-1/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEBN1_/custom_batch_gammaBlayer_with_weights-1/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEBN1_/custom_batch_moving_meanHlayer_with_weights-1/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!BN1_/custom_batch_moving_varianceLlayer_with_weights-1/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

Єtrace_0* 

Ѕtrace_0* 
* 
* 
* 

Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

Ћtrace_0* 

Ќtrace_0* 

L0
M1*

L0
M1*
* 

­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

Вtrace_0* 

Гtrace_0* 
\V
VARIABLE_VALUECNN2_/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN2_/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
U0
V1
W2
X3*

U0
V1*
* 

Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

Йtrace_0
Кtrace_1* 

Лtrace_0
Мtrace_1* 
qk
VARIABLE_VALUEBN2_/custom_batch_betaAlayer_with_weights-3/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEBN2_/custom_batch_gammaBlayer_with_weights-3/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEBN2_/custom_batch_moving_meanHlayer_with_weights-3/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!BN2_/custom_batch_moving_varianceLlayer_with_weights-3/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

Тtrace_0* 

Уtrace_0* 
* 
* 
* 

Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

Щtrace_0* 

Ъtrace_0* 

k0
l1*

k0
l1*
* 

Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

аtrace_0* 

бtrace_0* 
\V
VARIABLE_VALUECNN3_/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN3_/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
t0
u1
v2
w3*

t0
u1*
* 

вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

зtrace_0
иtrace_1* 

йtrace_0
кtrace_1* 
qk
VARIABLE_VALUEBN3_/custom_batch_betaAlayer_with_weights-5/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEBN3_/custom_batch_gammaBlayer_with_weights-5/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEBN3_/custom_batch_moving_meanHlayer_with_weights-5/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!BN3_/custom_batch_moving_varianceLlayer_with_weights-5/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 

рtrace_0* 

сtrace_0* 
* 
* 
* 

тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

чtrace_0* 

шtrace_0* 

0
1*

0
1*
* 

щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

юtrace_0* 

яtrace_0* 
\V
VARIABLE_VALUECNN4_/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN4_/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
0
1
2
3*

0
1*
* 

№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ѕtrace_0
іtrace_1* 

їtrace_0
јtrace_1* 
qk
VARIABLE_VALUEBN4_/custom_batch_betaAlayer_with_weights-7/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEBN4_/custom_batch_gammaBlayer_with_weights-7/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEBN4_/custom_batch_moving_meanHlayer_with_weights-7/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!BN4_/custom_batch_moving_varianceLlayer_with_weights-7/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

ўtrace_0* 

џtrace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

Љ0
Њ1*

Љ0
Њ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses*

trace_0* 

trace_0* 
\V
VARIABLE_VALUECNN5_/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN5_/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
В0
Г1
Д2
Е3*

В0
Г1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
qk
VARIABLE_VALUEBN5_/custom_batch_betaAlayer_with_weights-9/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEBN5_/custom_batch_gammaBlayer_with_weights-9/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEBN5_/custom_batch_moving_meanHlayer_with_weights-9/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!BN5_/custom_batch_moving_varianceLlayer_with_weights-9/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses* 

Ѓtrace_0* 

Єtrace_0* 

Ш0
Щ1*

Ш0
Щ1*
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses*

Њtrace_0* 

Ћtrace_0* 
]W
VARIABLE_VALUECNN6_/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
CNN6_/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
б0
в1
г2
д3*

б0
в1*
* 

Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses*

Бtrace_0
Вtrace_1* 

Гtrace_0
Дtrace_1* 
rl
VARIABLE_VALUEBN6_/custom_batch_betaBlayer_with_weights-11/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEBN6_/custom_batch_gammaClayer_with_weights-11/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEBN6_/custom_batch_moving_meanIlayer_with_weights-11/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!BN6_/custom_batch_moving_varianceMlayer_with_weights-11/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses* 

Кtrace_0* 

Лtrace_0* 
* 
* 
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
л	variables
мtrainable_variables
нregularization_losses
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses* 

Сtrace_0* 

Тtrace_0* 
* 
* 
* 

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses* 

Шtrace_0* 

Щtrace_0* 

э0
ю1*

э0
ю1*
* 

Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses*

Яtrace_0* 

аtrace_0* 
\V
VARIABLE_VALUEFC1_/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE	FC1_/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses* 

жtrace_0* 

зtrace_0* 
* 
* 
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 
`
80
91
W2
X3
v4
w5
6
7
Д8
Е9
г10
д11*
т
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28*

п0
р1*
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
э
0
с1
т2
у3
ф4
х5
ц6
ч7
ш8
щ9
ъ10
ы11
ь12
э13
ю14
я15
№16
ё17
ђ18
ѓ19
є20
ѕ21
і22
ї23
ј24
љ25
њ26*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
ф
с0
т1
у2
ф3
х4
ц5
ч6
ш7
щ8
ъ9
ы10
ь11
э12
ю13
я14
№15
ё16
ђ17
ѓ18
є19
ѕ20
і21
ї22
ј23
љ24
њ25*
ў
ћtrace_0
ќtrace_1
§trace_2
ўtrace_3
џtrace_4
trace_5
trace_6
trace_7
trace_8
trace_9
trace_10
trace_11
trace_12
trace_13
trace_14
trace_15
trace_16
trace_17
trace_18
trace_19
trace_20
trace_21
trace_22
trace_23
trace_24
trace_25* 
* 
* 
* 
* 
* 
* 
* 
* 

80
91*
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

W0
X1*
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

v0
w1*
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

0
1*
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

Д0
Е1*
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

г0
д1*
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
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
]W
VARIABLE_VALUESGD/m/CNN1_/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUESGD/m/CNN1_/bias1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUESGD/m/BN1_/custom_batch_beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUESGD/m/BN1_/custom_batch_gamma1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/m/CNN2_/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUESGD/m/CNN2_/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUESGD/m/BN2_/custom_batch_beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUESGD/m/BN2_/custom_batch_gamma1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/m/CNN3_/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUESGD/m/CNN3_/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUESGD/m/BN3_/custom_batch_beta2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUESGD/m/BN3_/custom_batch_gamma2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUESGD/m/CNN4_/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUESGD/m/CNN4_/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUESGD/m/BN4_/custom_batch_beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUESGD/m/BN4_/custom_batch_gamma2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUESGD/m/CNN5_/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUESGD/m/CNN5_/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUESGD/m/BN5_/custom_batch_beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUESGD/m/BN5_/custom_batch_gamma2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUESGD/m/CNN6_/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUESGD/m/CNN6_/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUESGD/m/BN6_/custom_batch_beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUESGD/m/BN6_/custom_batch_gamma2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/m/FC1_/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUESGD/m/FC1_/bias2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
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

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ё
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename CNN1_/kernel/Read/ReadVariableOpCNN1_/bias/Read/ReadVariableOp*BN1_/custom_batch_beta/Read/ReadVariableOp+BN1_/custom_batch_gamma/Read/ReadVariableOp1BN1_/custom_batch_moving_mean/Read/ReadVariableOp5BN1_/custom_batch_moving_variance/Read/ReadVariableOp CNN2_/kernel/Read/ReadVariableOpCNN2_/bias/Read/ReadVariableOp*BN2_/custom_batch_beta/Read/ReadVariableOp+BN2_/custom_batch_gamma/Read/ReadVariableOp1BN2_/custom_batch_moving_mean/Read/ReadVariableOp5BN2_/custom_batch_moving_variance/Read/ReadVariableOp CNN3_/kernel/Read/ReadVariableOpCNN3_/bias/Read/ReadVariableOp*BN3_/custom_batch_beta/Read/ReadVariableOp+BN3_/custom_batch_gamma/Read/ReadVariableOp1BN3_/custom_batch_moving_mean/Read/ReadVariableOp5BN3_/custom_batch_moving_variance/Read/ReadVariableOp CNN4_/kernel/Read/ReadVariableOpCNN4_/bias/Read/ReadVariableOp*BN4_/custom_batch_beta/Read/ReadVariableOp+BN4_/custom_batch_gamma/Read/ReadVariableOp1BN4_/custom_batch_moving_mean/Read/ReadVariableOp5BN4_/custom_batch_moving_variance/Read/ReadVariableOp CNN5_/kernel/Read/ReadVariableOpCNN5_/bias/Read/ReadVariableOp*BN5_/custom_batch_beta/Read/ReadVariableOp+BN5_/custom_batch_gamma/Read/ReadVariableOp1BN5_/custom_batch_moving_mean/Read/ReadVariableOp5BN5_/custom_batch_moving_variance/Read/ReadVariableOp CNN6_/kernel/Read/ReadVariableOpCNN6_/bias/Read/ReadVariableOp*BN6_/custom_batch_beta/Read/ReadVariableOp+BN6_/custom_batch_gamma/Read/ReadVariableOp1BN6_/custom_batch_moving_mean/Read/ReadVariableOp5BN6_/custom_batch_moving_variance/Read/ReadVariableOpFC1_/kernel/Read/ReadVariableOpFC1_/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp&SGD/m/CNN1_/kernel/Read/ReadVariableOp$SGD/m/CNN1_/bias/Read/ReadVariableOp0SGD/m/BN1_/custom_batch_beta/Read/ReadVariableOp1SGD/m/BN1_/custom_batch_gamma/Read/ReadVariableOp&SGD/m/CNN2_/kernel/Read/ReadVariableOp$SGD/m/CNN2_/bias/Read/ReadVariableOp0SGD/m/BN2_/custom_batch_beta/Read/ReadVariableOp1SGD/m/BN2_/custom_batch_gamma/Read/ReadVariableOp&SGD/m/CNN3_/kernel/Read/ReadVariableOp$SGD/m/CNN3_/bias/Read/ReadVariableOp0SGD/m/BN3_/custom_batch_beta/Read/ReadVariableOp1SGD/m/BN3_/custom_batch_gamma/Read/ReadVariableOp&SGD/m/CNN4_/kernel/Read/ReadVariableOp$SGD/m/CNN4_/bias/Read/ReadVariableOp0SGD/m/BN4_/custom_batch_beta/Read/ReadVariableOp1SGD/m/BN4_/custom_batch_gamma/Read/ReadVariableOp&SGD/m/CNN5_/kernel/Read/ReadVariableOp$SGD/m/CNN5_/bias/Read/ReadVariableOp0SGD/m/BN5_/custom_batch_beta/Read/ReadVariableOp1SGD/m/BN5_/custom_batch_gamma/Read/ReadVariableOp&SGD/m/CNN6_/kernel/Read/ReadVariableOp$SGD/m/CNN6_/bias/Read/ReadVariableOp0SGD/m/BN6_/custom_batch_beta/Read/ReadVariableOp1SGD/m/BN6_/custom_batch_gamma/Read/ReadVariableOp%SGD/m/FC1_/kernel/Read/ReadVariableOp#SGD/m/FC1_/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*S
TinL
J2H	*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_17833
є
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameCNN1_/kernel
CNN1_/biasBN1_/custom_batch_betaBN1_/custom_batch_gammaBN1_/custom_batch_moving_mean!BN1_/custom_batch_moving_varianceCNN2_/kernel
CNN2_/biasBN2_/custom_batch_betaBN2_/custom_batch_gammaBN2_/custom_batch_moving_mean!BN2_/custom_batch_moving_varianceCNN3_/kernel
CNN3_/biasBN3_/custom_batch_betaBN3_/custom_batch_gammaBN3_/custom_batch_moving_mean!BN3_/custom_batch_moving_varianceCNN4_/kernel
CNN4_/biasBN4_/custom_batch_betaBN4_/custom_batch_gammaBN4_/custom_batch_moving_mean!BN4_/custom_batch_moving_varianceCNN5_/kernel
CNN5_/biasBN5_/custom_batch_betaBN5_/custom_batch_gammaBN5_/custom_batch_moving_mean!BN5_/custom_batch_moving_varianceCNN6_/kernel
CNN6_/biasBN6_/custom_batch_betaBN6_/custom_batch_gammaBN6_/custom_batch_moving_mean!BN6_/custom_batch_moving_varianceFC1_/kernel	FC1_/bias	iterationlearning_rateSGD/m/CNN1_/kernelSGD/m/CNN1_/biasSGD/m/BN1_/custom_batch_betaSGD/m/BN1_/custom_batch_gammaSGD/m/CNN2_/kernelSGD/m/CNN2_/biasSGD/m/BN2_/custom_batch_betaSGD/m/BN2_/custom_batch_gammaSGD/m/CNN3_/kernelSGD/m/CNN3_/biasSGD/m/BN3_/custom_batch_betaSGD/m/BN3_/custom_batch_gammaSGD/m/CNN4_/kernelSGD/m/CNN4_/biasSGD/m/BN4_/custom_batch_betaSGD/m/BN4_/custom_batch_gammaSGD/m/CNN5_/kernelSGD/m/CNN5_/biasSGD/m/BN5_/custom_batch_betaSGD/m/BN5_/custom_batch_gammaSGD/m/CNN6_/kernelSGD/m/CNN6_/biasSGD/m/BN6_/custom_batch_betaSGD/m/BN6_/custom_batch_gammaSGD/m/FC1_/kernelSGD/m/FC1_/biastotal_1count_1totalcount*R
TinK
I2G*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_18053щ
Ћ
J
"__inference__update_step_xla_16776
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
K
/__inference_max_pooling2d_2_layer_call_fn_17175

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_14438
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_14414

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ї
П
$__inference_BN6__layer_call_fn_17458

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN6__layer_call_and_return_conditional_losses_14784w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ `: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
с
љ
%__inference_model_layer_call_fn_16255

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 0

unknown_18:0

unknown_19:0

unknown_20:0

unknown_21:0

unknown_22:0$

unknown_23:0@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@`

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:
identityЂStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*<
_read_only_resource_inputs
 #$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17180

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р
C
'__inference_re_lu_3_layer_call_fn_17288

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_14693i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ0:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ј

љ
@__inference_CNN4__layer_call_and_return_conditional_losses_17199

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Р
C
'__inference_re_lu_1_layer_call_fn_17042

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_14587i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё*
Ъ
?__inference_BN1__layer_call_and_return_conditional_losses_15420

inputs%
readvariableop_resource:'
readvariableop_2_resource:'
readvariableop_4_resource:+
add_3_readvariableop_resource:
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџL
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ж
?__inference_BN1__layer_call_and_return_conditional_losses_14519

inputs)
sub_readvariableop_resource:%
readvariableop_resource:'
readvariableop_1_resource:+
add_1_readvariableop_resource:
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
::
SqrtSqrtadd:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ

љ
@__inference_CNN6__layer_call_and_return_conditional_losses_14759

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ @
 
_user_specified_nameinputs
Љ
П
$__inference_BN4__layer_call_fn_17225

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN4__layer_call_and_return_conditional_losses_15174x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16701
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

ж
?__inference_BN3__layer_call_and_return_conditional_losses_17121

inputs)
sub_readvariableop_resource: %
readvariableop_resource: '
readvariableop_1_resource: +
add_1_readvariableop_resource: 
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
: :
SqrtSqrtadd:z:0*
T0*
_output_shapes
: `
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ a
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Й
K
/__inference_max_pooling2d_1_layer_call_fn_17052

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_14426
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ж
?__inference_BN2__layer_call_and_return_conditional_losses_16998

inputs)
sub_readvariableop_resource:%
readvariableop_resource:'
readvariableop_1_resource:+
add_1_readvariableop_resource:
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
::
SqrtSqrtadd:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 б
н
@__inference_model_layer_call_and_return_conditional_losses_16681

inputs>
$cnn1__conv2d_readvariableop_resource:3
%cnn1__biasadd_readvariableop_resource:*
bn1__readvariableop_resource:,
bn1__readvariableop_2_resource:,
bn1__readvariableop_4_resource:0
"bn1__add_3_readvariableop_resource:>
$cnn2__conv2d_readvariableop_resource:3
%cnn2__biasadd_readvariableop_resource:*
bn2__readvariableop_resource:,
bn2__readvariableop_2_resource:,
bn2__readvariableop_4_resource:0
"bn2__add_3_readvariableop_resource:>
$cnn3__conv2d_readvariableop_resource: 3
%cnn3__biasadd_readvariableop_resource: *
bn3__readvariableop_resource: ,
bn3__readvariableop_2_resource: ,
bn3__readvariableop_4_resource: 0
"bn3__add_3_readvariableop_resource: >
$cnn4__conv2d_readvariableop_resource: 03
%cnn4__biasadd_readvariableop_resource:0*
bn4__readvariableop_resource:0,
bn4__readvariableop_2_resource:0,
bn4__readvariableop_4_resource:00
"bn4__add_3_readvariableop_resource:0>
$cnn5__conv2d_readvariableop_resource:0@3
%cnn5__biasadd_readvariableop_resource:@*
bn5__readvariableop_resource:@,
bn5__readvariableop_2_resource:@,
bn5__readvariableop_4_resource:@0
"bn5__add_3_readvariableop_resource:@>
$cnn6__conv2d_readvariableop_resource:@`3
%cnn6__biasadd_readvariableop_resource:`*
bn6__readvariableop_resource:`,
bn6__readvariableop_2_resource:`,
bn6__readvariableop_4_resource:`0
"bn6__add_3_readvariableop_resource:`5
#fc1__matmul_readvariableop_resource:`2
$fc1__biasadd_readvariableop_resource:
identityЂBN1_/AssignVariableOpЂBN1_/AssignVariableOp_1ЂBN1_/AssignVariableOp_2ЂBN1_/AssignVariableOp_3ЂBN1_/ReadVariableOpЂBN1_/ReadVariableOp_1ЂBN1_/ReadVariableOp_2ЂBN1_/ReadVariableOp_3ЂBN1_/ReadVariableOp_4ЂBN1_/add_3/ReadVariableOpЂBN2_/AssignVariableOpЂBN2_/AssignVariableOp_1ЂBN2_/AssignVariableOp_2ЂBN2_/AssignVariableOp_3ЂBN2_/ReadVariableOpЂBN2_/ReadVariableOp_1ЂBN2_/ReadVariableOp_2ЂBN2_/ReadVariableOp_3ЂBN2_/ReadVariableOp_4ЂBN2_/add_3/ReadVariableOpЂBN3_/AssignVariableOpЂBN3_/AssignVariableOp_1ЂBN3_/AssignVariableOp_2ЂBN3_/AssignVariableOp_3ЂBN3_/ReadVariableOpЂBN3_/ReadVariableOp_1ЂBN3_/ReadVariableOp_2ЂBN3_/ReadVariableOp_3ЂBN3_/ReadVariableOp_4ЂBN3_/add_3/ReadVariableOpЂBN4_/AssignVariableOpЂBN4_/AssignVariableOp_1ЂBN4_/AssignVariableOp_2ЂBN4_/AssignVariableOp_3ЂBN4_/ReadVariableOpЂBN4_/ReadVariableOp_1ЂBN4_/ReadVariableOp_2ЂBN4_/ReadVariableOp_3ЂBN4_/ReadVariableOp_4ЂBN4_/add_3/ReadVariableOpЂBN5_/AssignVariableOpЂBN5_/AssignVariableOp_1ЂBN5_/AssignVariableOp_2ЂBN5_/AssignVariableOp_3ЂBN5_/ReadVariableOpЂBN5_/ReadVariableOp_1ЂBN5_/ReadVariableOp_2ЂBN5_/ReadVariableOp_3ЂBN5_/ReadVariableOp_4ЂBN5_/add_3/ReadVariableOpЂBN6_/AssignVariableOpЂBN6_/AssignVariableOp_1ЂBN6_/AssignVariableOp_2ЂBN6_/AssignVariableOp_3ЂBN6_/ReadVariableOpЂBN6_/ReadVariableOp_1ЂBN6_/ReadVariableOp_2ЂBN6_/ReadVariableOp_3ЂBN6_/ReadVariableOp_4ЂBN6_/add_3/ReadVariableOpЂCNN1_/BiasAdd/ReadVariableOpЂCNN1_/Conv2D/ReadVariableOpЂCNN2_/BiasAdd/ReadVariableOpЂCNN2_/Conv2D/ReadVariableOpЂCNN3_/BiasAdd/ReadVariableOpЂCNN3_/Conv2D/ReadVariableOpЂCNN4_/BiasAdd/ReadVariableOpЂCNN4_/Conv2D/ReadVariableOpЂCNN5_/BiasAdd/ReadVariableOpЂCNN5_/Conv2D/ReadVariableOpЂCNN6_/BiasAdd/ReadVariableOpЂCNN6_/Conv2D/ReadVariableOpЂFC1_/BiasAdd/ReadVariableOpЂFC1_/MatMul/ReadVariableOp
CNN1_/Conv2D/ReadVariableOpReadVariableOp$cnn1__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0І
CNN1_/Conv2DConv2Dinputs#CNN1_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
~
CNN1_/BiasAdd/ReadVariableOpReadVariableOp%cnn1__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
CNN1_/BiasAddBiasAddCNN1_/Conv2D:output:0$CNN1_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџx
#BN1_/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
BN1_/moments/meanMeanCNN1_/BiasAdd:output:0,BN1_/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(v
BN1_/moments/StopGradientStopGradientBN1_/moments/mean:output:0*
T0*&
_output_shapes
:Њ
BN1_/moments/SquaredDifferenceSquaredDifferenceCNN1_/BiasAdd:output:0"BN1_/moments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ|
'BN1_/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Е
BN1_/moments/varianceMean"BN1_/moments/SquaredDifference:z:00BN1_/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(y
BN1_/moments/SqueezeSqueezeBN1_/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
BN1_/moments/Squeeze_1SqueezeBN1_/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 l
BN1_/ReadVariableOpReadVariableOpbn1__readvariableop_resource*
_output_shapes
:*
dtype0O

BN1_/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?f
BN1_/mulMulBN1_/ReadVariableOp:value:0BN1_/mul/y:output:0*
T0*
_output_shapes
:Q
BN1_/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=l

BN1_/mul_1MulBN1_/moments/Squeeze:output:0BN1_/mul_1/y:output:0*
T0*
_output_shapes
:T
BN1_/addAddV2BN1_/mul:z:0BN1_/mul_1:z:0*
T0*
_output_shapes
:Ь
BN1_/AssignVariableOpAssignVariableOpbn1__readvariableop_resourceBN1_/add:z:0^BN1_/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN1_/ReadVariableOp_1ReadVariableOpbn1__readvariableop_resource^BN1_/AssignVariableOp*
_output_shapes
:*
dtype0б
BN1_/AssignVariableOp_1AssignVariableOpbn1__readvariableop_resourceBN1_/ReadVariableOp_1:value:0^BN1_/AssignVariableOp^BN1_/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
BN1_/ReadVariableOp_2ReadVariableOpbn1__readvariableop_2_resource*
_output_shapes
:*
dtype0Q
BN1_/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?l

BN1_/mul_2MulBN1_/ReadVariableOp_2:value:0BN1_/mul_2/y:output:0*
T0*
_output_shapes
:Q
BN1_/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=n

BN1_/mul_3MulBN1_/moments/Squeeze_1:output:0BN1_/mul_3/y:output:0*
T0*
_output_shapes
:X

BN1_/add_1AddV2BN1_/mul_2:z:0BN1_/mul_3:z:0*
T0*
_output_shapes
:д
BN1_/AssignVariableOp_2AssignVariableOpbn1__readvariableop_2_resourceBN1_/add_1:z:0^BN1_/ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN1_/ReadVariableOp_3ReadVariableOpbn1__readvariableop_2_resource^BN1_/AssignVariableOp_2*
_output_shapes
:*
dtype0е
BN1_/AssignVariableOp_3AssignVariableOpbn1__readvariableop_2_resourceBN1_/ReadVariableOp_3:value:0^BN1_/AssignVariableOp_2^BN1_/ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(
BN1_/subSubCNN1_/BiasAdd:output:0BN1_/moments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџQ
BN1_/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75p

BN1_/add_2AddV2BN1_/moments/Squeeze_1:output:0BN1_/add_2/y:output:0*
T0*
_output_shapes
:F
	BN1_/SqrtSqrtBN1_/add_2:z:0*
T0*
_output_shapes
:o
BN1_/truedivRealDivBN1_/sub:z:0BN1_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџp
BN1_/ReadVariableOp_4ReadVariableOpbn1__readvariableop_4_resource*
_output_shapes
:*
dtype0}

BN1_/mul_4MulBN1_/ReadVariableOp_4:value:0BN1_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџx
BN1_/add_3/ReadVariableOpReadVariableOp"bn1__add_3_readvariableop_resource*
_output_shapes
:*
dtype0

BN1_/add_3AddV2BN1_/mul_4:z:0!BN1_/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ]

re_lu/ReluReluBN1_/add_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџЈ
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

CNN2_/Conv2D/ReadVariableOpReadVariableOp$cnn2__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0О
CNN2_/Conv2DConv2Dmax_pooling2d/MaxPool:output:0#CNN2_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
~
CNN2_/BiasAdd/ReadVariableOpReadVariableOp%cnn2__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
CNN2_/BiasAddBiasAddCNN2_/Conv2D:output:0$CNN2_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџx
#BN2_/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
BN2_/moments/meanMeanCNN2_/BiasAdd:output:0,BN2_/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(v
BN2_/moments/StopGradientStopGradientBN2_/moments/mean:output:0*
T0*&
_output_shapes
:Њ
BN2_/moments/SquaredDifferenceSquaredDifferenceCNN2_/BiasAdd:output:0"BN2_/moments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ|
'BN2_/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Е
BN2_/moments/varianceMean"BN2_/moments/SquaredDifference:z:00BN2_/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(y
BN2_/moments/SqueezeSqueezeBN2_/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
BN2_/moments/Squeeze_1SqueezeBN2_/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 l
BN2_/ReadVariableOpReadVariableOpbn2__readvariableop_resource*
_output_shapes
:*
dtype0O

BN2_/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?f
BN2_/mulMulBN2_/ReadVariableOp:value:0BN2_/mul/y:output:0*
T0*
_output_shapes
:Q
BN2_/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=l

BN2_/mul_1MulBN2_/moments/Squeeze:output:0BN2_/mul_1/y:output:0*
T0*
_output_shapes
:T
BN2_/addAddV2BN2_/mul:z:0BN2_/mul_1:z:0*
T0*
_output_shapes
:Ь
BN2_/AssignVariableOpAssignVariableOpbn2__readvariableop_resourceBN2_/add:z:0^BN2_/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN2_/ReadVariableOp_1ReadVariableOpbn2__readvariableop_resource^BN2_/AssignVariableOp*
_output_shapes
:*
dtype0б
BN2_/AssignVariableOp_1AssignVariableOpbn2__readvariableop_resourceBN2_/ReadVariableOp_1:value:0^BN2_/AssignVariableOp^BN2_/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
BN2_/ReadVariableOp_2ReadVariableOpbn2__readvariableop_2_resource*
_output_shapes
:*
dtype0Q
BN2_/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?l

BN2_/mul_2MulBN2_/ReadVariableOp_2:value:0BN2_/mul_2/y:output:0*
T0*
_output_shapes
:Q
BN2_/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=n

BN2_/mul_3MulBN2_/moments/Squeeze_1:output:0BN2_/mul_3/y:output:0*
T0*
_output_shapes
:X

BN2_/add_1AddV2BN2_/mul_2:z:0BN2_/mul_3:z:0*
T0*
_output_shapes
:д
BN2_/AssignVariableOp_2AssignVariableOpbn2__readvariableop_2_resourceBN2_/add_1:z:0^BN2_/ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN2_/ReadVariableOp_3ReadVariableOpbn2__readvariableop_2_resource^BN2_/AssignVariableOp_2*
_output_shapes
:*
dtype0е
BN2_/AssignVariableOp_3AssignVariableOpbn2__readvariableop_2_resourceBN2_/ReadVariableOp_3:value:0^BN2_/AssignVariableOp_2^BN2_/ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(
BN2_/subSubCNN2_/BiasAdd:output:0BN2_/moments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџQ
BN2_/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75p

BN2_/add_2AddV2BN2_/moments/Squeeze_1:output:0BN2_/add_2/y:output:0*
T0*
_output_shapes
:F
	BN2_/SqrtSqrtBN2_/add_2:z:0*
T0*
_output_shapes
:o
BN2_/truedivRealDivBN2_/sub:z:0BN2_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџp
BN2_/ReadVariableOp_4ReadVariableOpbn2__readvariableop_4_resource*
_output_shapes
:*
dtype0}

BN2_/mul_4MulBN2_/ReadVariableOp_4:value:0BN2_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџx
BN2_/add_3/ReadVariableOpReadVariableOp"bn2__add_3_readvariableop_resource*
_output_shapes
:*
dtype0

BN2_/add_3AddV2BN2_/mul_4:z:0!BN2_/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ_
re_lu_1/ReluReluBN2_/add_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџЌ
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

CNN3_/Conv2D/ReadVariableOpReadVariableOp$cnn3__conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Р
CNN3_/Conv2DConv2D max_pooling2d_1/MaxPool:output:0#CNN3_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
~
CNN3_/BiasAdd/ReadVariableOpReadVariableOp%cnn3__biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
CNN3_/BiasAddBiasAddCNN3_/Conv2D:output:0$CNN3_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ x
#BN3_/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
BN3_/moments/meanMeanCNN3_/BiasAdd:output:0,BN3_/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(v
BN3_/moments/StopGradientStopGradientBN3_/moments/mean:output:0*
T0*&
_output_shapes
: Њ
BN3_/moments/SquaredDifferenceSquaredDifferenceCNN3_/BiasAdd:output:0"BN3_/moments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ |
'BN3_/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Е
BN3_/moments/varianceMean"BN3_/moments/SquaredDifference:z:00BN3_/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(y
BN3_/moments/SqueezeSqueezeBN3_/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 
BN3_/moments/Squeeze_1SqueezeBN3_/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 l
BN3_/ReadVariableOpReadVariableOpbn3__readvariableop_resource*
_output_shapes
: *
dtype0O

BN3_/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?f
BN3_/mulMulBN3_/ReadVariableOp:value:0BN3_/mul/y:output:0*
T0*
_output_shapes
: Q
BN3_/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=l

BN3_/mul_1MulBN3_/moments/Squeeze:output:0BN3_/mul_1/y:output:0*
T0*
_output_shapes
: T
BN3_/addAddV2BN3_/mul:z:0BN3_/mul_1:z:0*
T0*
_output_shapes
: Ь
BN3_/AssignVariableOpAssignVariableOpbn3__readvariableop_resourceBN3_/add:z:0^BN3_/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN3_/ReadVariableOp_1ReadVariableOpbn3__readvariableop_resource^BN3_/AssignVariableOp*
_output_shapes
: *
dtype0б
BN3_/AssignVariableOp_1AssignVariableOpbn3__readvariableop_resourceBN3_/ReadVariableOp_1:value:0^BN3_/AssignVariableOp^BN3_/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
BN3_/ReadVariableOp_2ReadVariableOpbn3__readvariableop_2_resource*
_output_shapes
: *
dtype0Q
BN3_/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?l

BN3_/mul_2MulBN3_/ReadVariableOp_2:value:0BN3_/mul_2/y:output:0*
T0*
_output_shapes
: Q
BN3_/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=n

BN3_/mul_3MulBN3_/moments/Squeeze_1:output:0BN3_/mul_3/y:output:0*
T0*
_output_shapes
: X

BN3_/add_1AddV2BN3_/mul_2:z:0BN3_/mul_3:z:0*
T0*
_output_shapes
: д
BN3_/AssignVariableOp_2AssignVariableOpbn3__readvariableop_2_resourceBN3_/add_1:z:0^BN3_/ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN3_/ReadVariableOp_3ReadVariableOpbn3__readvariableop_2_resource^BN3_/AssignVariableOp_2*
_output_shapes
: *
dtype0е
BN3_/AssignVariableOp_3AssignVariableOpbn3__readvariableop_2_resourceBN3_/ReadVariableOp_3:value:0^BN3_/AssignVariableOp_2^BN3_/ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(
BN3_/subSubCNN3_/BiasAdd:output:0BN3_/moments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџ Q
BN3_/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75p

BN3_/add_2AddV2BN3_/moments/Squeeze_1:output:0BN3_/add_2/y:output:0*
T0*
_output_shapes
: F
	BN3_/SqrtSqrtBN3_/add_2:z:0*
T0*
_output_shapes
: o
BN3_/truedivRealDivBN3_/sub:z:0BN3_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ p
BN3_/ReadVariableOp_4ReadVariableOpbn3__readvariableop_4_resource*
_output_shapes
: *
dtype0}

BN3_/mul_4MulBN3_/ReadVariableOp_4:value:0BN3_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ x
BN3_/add_3/ReadVariableOpReadVariableOp"bn3__add_3_readvariableop_resource*
_output_shapes
: *
dtype0

BN3_/add_3AddV2BN3_/mul_4:z:0!BN3_/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ _
re_lu_2/ReluReluBN3_/add_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ Ќ
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides

CNN4_/Conv2D/ReadVariableOpReadVariableOp$cnn4__conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Р
CNN4_/Conv2DConv2D max_pooling2d_2/MaxPool:output:0#CNN4_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides
~
CNN4_/BiasAdd/ReadVariableOpReadVariableOp%cnn4__biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
CNN4_/BiasAddBiasAddCNN4_/Conv2D:output:0$CNN4_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0x
#BN4_/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
BN4_/moments/meanMeanCNN4_/BiasAdd:output:0,BN4_/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:0*
	keep_dims(v
BN4_/moments/StopGradientStopGradientBN4_/moments/mean:output:0*
T0*&
_output_shapes
:0Њ
BN4_/moments/SquaredDifferenceSquaredDifferenceCNN4_/BiasAdd:output:0"BN4_/moments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0|
'BN4_/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Е
BN4_/moments/varianceMean"BN4_/moments/SquaredDifference:z:00BN4_/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:0*
	keep_dims(y
BN4_/moments/SqueezeSqueezeBN4_/moments/mean:output:0*
T0*
_output_shapes
:0*
squeeze_dims
 
BN4_/moments/Squeeze_1SqueezeBN4_/moments/variance:output:0*
T0*
_output_shapes
:0*
squeeze_dims
 l
BN4_/ReadVariableOpReadVariableOpbn4__readvariableop_resource*
_output_shapes
:0*
dtype0O

BN4_/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?f
BN4_/mulMulBN4_/ReadVariableOp:value:0BN4_/mul/y:output:0*
T0*
_output_shapes
:0Q
BN4_/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=l

BN4_/mul_1MulBN4_/moments/Squeeze:output:0BN4_/mul_1/y:output:0*
T0*
_output_shapes
:0T
BN4_/addAddV2BN4_/mul:z:0BN4_/mul_1:z:0*
T0*
_output_shapes
:0Ь
BN4_/AssignVariableOpAssignVariableOpbn4__readvariableop_resourceBN4_/add:z:0^BN4_/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN4_/ReadVariableOp_1ReadVariableOpbn4__readvariableop_resource^BN4_/AssignVariableOp*
_output_shapes
:0*
dtype0б
BN4_/AssignVariableOp_1AssignVariableOpbn4__readvariableop_resourceBN4_/ReadVariableOp_1:value:0^BN4_/AssignVariableOp^BN4_/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
BN4_/ReadVariableOp_2ReadVariableOpbn4__readvariableop_2_resource*
_output_shapes
:0*
dtype0Q
BN4_/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?l

BN4_/mul_2MulBN4_/ReadVariableOp_2:value:0BN4_/mul_2/y:output:0*
T0*
_output_shapes
:0Q
BN4_/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=n

BN4_/mul_3MulBN4_/moments/Squeeze_1:output:0BN4_/mul_3/y:output:0*
T0*
_output_shapes
:0X

BN4_/add_1AddV2BN4_/mul_2:z:0BN4_/mul_3:z:0*
T0*
_output_shapes
:0д
BN4_/AssignVariableOp_2AssignVariableOpbn4__readvariableop_2_resourceBN4_/add_1:z:0^BN4_/ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN4_/ReadVariableOp_3ReadVariableOpbn4__readvariableop_2_resource^BN4_/AssignVariableOp_2*
_output_shapes
:0*
dtype0е
BN4_/AssignVariableOp_3AssignVariableOpbn4__readvariableop_2_resourceBN4_/ReadVariableOp_3:value:0^BN4_/AssignVariableOp_2^BN4_/ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(
BN4_/subSubCNN4_/BiasAdd:output:0BN4_/moments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0Q
BN4_/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75p

BN4_/add_2AddV2BN4_/moments/Squeeze_1:output:0BN4_/add_2/y:output:0*
T0*
_output_shapes
:0F
	BN4_/SqrtSqrtBN4_/add_2:z:0*
T0*
_output_shapes
:0o
BN4_/truedivRealDivBN4_/sub:z:0BN4_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ0p
BN4_/ReadVariableOp_4ReadVariableOpbn4__readvariableop_4_resource*
_output_shapes
:0*
dtype0}

BN4_/mul_4MulBN4_/ReadVariableOp_4:value:0BN4_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0x
BN4_/add_3/ReadVariableOpReadVariableOp"bn4__add_3_readvariableop_resource*
_output_shapes
:0*
dtype0

BN4_/add_3AddV2BN4_/mul_4:z:0!BN4_/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0_
re_lu_3/ReluReluBN4_/add_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0Ћ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@0*
ksize
*
paddingVALID*
strides

CNN5_/Conv2D/ReadVariableOpReadVariableOp$cnn5__conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0П
CNN5_/Conv2DConv2D max_pooling2d_3/MaxPool:output:0#CNN5_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
~
CNN5_/BiasAdd/ReadVariableOpReadVariableOp%cnn5__biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
CNN5_/BiasAddBiasAddCNN5_/Conv2D:output:0$CNN5_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@x
#BN5_/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
BN5_/moments/meanMeanCNN5_/BiasAdd:output:0,BN5_/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(v
BN5_/moments/StopGradientStopGradientBN5_/moments/mean:output:0*
T0*&
_output_shapes
:@Љ
BN5_/moments/SquaredDifferenceSquaredDifferenceCNN5_/BiasAdd:output:0"BN5_/moments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@|
'BN5_/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Е
BN5_/moments/varianceMean"BN5_/moments/SquaredDifference:z:00BN5_/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(y
BN5_/moments/SqueezeSqueezeBN5_/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 
BN5_/moments/Squeeze_1SqueezeBN5_/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 l
BN5_/ReadVariableOpReadVariableOpbn5__readvariableop_resource*
_output_shapes
:@*
dtype0O

BN5_/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?f
BN5_/mulMulBN5_/ReadVariableOp:value:0BN5_/mul/y:output:0*
T0*
_output_shapes
:@Q
BN5_/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=l

BN5_/mul_1MulBN5_/moments/Squeeze:output:0BN5_/mul_1/y:output:0*
T0*
_output_shapes
:@T
BN5_/addAddV2BN5_/mul:z:0BN5_/mul_1:z:0*
T0*
_output_shapes
:@Ь
BN5_/AssignVariableOpAssignVariableOpbn5__readvariableop_resourceBN5_/add:z:0^BN5_/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN5_/ReadVariableOp_1ReadVariableOpbn5__readvariableop_resource^BN5_/AssignVariableOp*
_output_shapes
:@*
dtype0б
BN5_/AssignVariableOp_1AssignVariableOpbn5__readvariableop_resourceBN5_/ReadVariableOp_1:value:0^BN5_/AssignVariableOp^BN5_/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
BN5_/ReadVariableOp_2ReadVariableOpbn5__readvariableop_2_resource*
_output_shapes
:@*
dtype0Q
BN5_/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?l

BN5_/mul_2MulBN5_/ReadVariableOp_2:value:0BN5_/mul_2/y:output:0*
T0*
_output_shapes
:@Q
BN5_/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=n

BN5_/mul_3MulBN5_/moments/Squeeze_1:output:0BN5_/mul_3/y:output:0*
T0*
_output_shapes
:@X

BN5_/add_1AddV2BN5_/mul_2:z:0BN5_/mul_3:z:0*
T0*
_output_shapes
:@д
BN5_/AssignVariableOp_2AssignVariableOpbn5__readvariableop_2_resourceBN5_/add_1:z:0^BN5_/ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN5_/ReadVariableOp_3ReadVariableOpbn5__readvariableop_2_resource^BN5_/AssignVariableOp_2*
_output_shapes
:@*
dtype0е
BN5_/AssignVariableOp_3AssignVariableOpbn5__readvariableop_2_resourceBN5_/ReadVariableOp_3:value:0^BN5_/AssignVariableOp_2^BN5_/ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(
BN5_/subSubCNN5_/BiasAdd:output:0BN5_/moments/Squeeze:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@Q
BN5_/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75p

BN5_/add_2AddV2BN5_/moments/Squeeze_1:output:0BN5_/add_2/y:output:0*
T0*
_output_shapes
:@F
	BN5_/SqrtSqrtBN5_/add_2:z:0*
T0*
_output_shapes
:@n
BN5_/truedivRealDivBN5_/sub:z:0BN5_/Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@p
BN5_/ReadVariableOp_4ReadVariableOpbn5__readvariableop_4_resource*
_output_shapes
:@*
dtype0|

BN5_/mul_4MulBN5_/ReadVariableOp_4:value:0BN5_/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@x
BN5_/add_3/ReadVariableOpReadVariableOp"bn5__add_3_readvariableop_resource*
_output_shapes
:@*
dtype0

BN5_/add_3AddV2BN5_/mul_4:z:0!BN5_/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@^
re_lu_4/ReluReluBN5_/add_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@Ћ
max_pooling2d_4/MaxPoolMaxPoolre_lu_4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides

CNN6_/Conv2D/ReadVariableOpReadVariableOp$cnn6__conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0П
CNN6_/Conv2DConv2D max_pooling2d_4/MaxPool:output:0#CNN6_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `*
paddingSAME*
strides
~
CNN6_/BiasAdd/ReadVariableOpReadVariableOp%cnn6__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
CNN6_/BiasAddBiasAddCNN6_/Conv2D:output:0$CNN6_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `x
#BN6_/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ё
BN6_/moments/meanMeanCNN6_/BiasAdd:output:0,BN6_/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:`*
	keep_dims(v
BN6_/moments/StopGradientStopGradientBN6_/moments/mean:output:0*
T0*&
_output_shapes
:`Љ
BN6_/moments/SquaredDifferenceSquaredDifferenceCNN6_/BiasAdd:output:0"BN6_/moments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџ `|
'BN6_/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Е
BN6_/moments/varianceMean"BN6_/moments/SquaredDifference:z:00BN6_/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:`*
	keep_dims(y
BN6_/moments/SqueezeSqueezeBN6_/moments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 
BN6_/moments/Squeeze_1SqueezeBN6_/moments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 l
BN6_/ReadVariableOpReadVariableOpbn6__readvariableop_resource*
_output_shapes
:`*
dtype0O

BN6_/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?f
BN6_/mulMulBN6_/ReadVariableOp:value:0BN6_/mul/y:output:0*
T0*
_output_shapes
:`Q
BN6_/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=l

BN6_/mul_1MulBN6_/moments/Squeeze:output:0BN6_/mul_1/y:output:0*
T0*
_output_shapes
:`T
BN6_/addAddV2BN6_/mul:z:0BN6_/mul_1:z:0*
T0*
_output_shapes
:`Ь
BN6_/AssignVariableOpAssignVariableOpbn6__readvariableop_resourceBN6_/add:z:0^BN6_/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN6_/ReadVariableOp_1ReadVariableOpbn6__readvariableop_resource^BN6_/AssignVariableOp*
_output_shapes
:`*
dtype0б
BN6_/AssignVariableOp_1AssignVariableOpbn6__readvariableop_resourceBN6_/ReadVariableOp_1:value:0^BN6_/AssignVariableOp^BN6_/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
BN6_/ReadVariableOp_2ReadVariableOpbn6__readvariableop_2_resource*
_output_shapes
:`*
dtype0Q
BN6_/mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?l

BN6_/mul_2MulBN6_/ReadVariableOp_2:value:0BN6_/mul_2/y:output:0*
T0*
_output_shapes
:`Q
BN6_/mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=n

BN6_/mul_3MulBN6_/moments/Squeeze_1:output:0BN6_/mul_3/y:output:0*
T0*
_output_shapes
:`X

BN6_/add_1AddV2BN6_/mul_2:z:0BN6_/mul_3:z:0*
T0*
_output_shapes
:`д
BN6_/AssignVariableOp_2AssignVariableOpbn6__readvariableop_2_resourceBN6_/add_1:z:0^BN6_/ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(
BN6_/ReadVariableOp_3ReadVariableOpbn6__readvariableop_2_resource^BN6_/AssignVariableOp_2*
_output_shapes
:`*
dtype0е
BN6_/AssignVariableOp_3AssignVariableOpbn6__readvariableop_2_resourceBN6_/ReadVariableOp_3:value:0^BN6_/AssignVariableOp_2^BN6_/ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(
BN6_/subSubCNN6_/BiasAdd:output:0BN6_/moments/Squeeze:output:0*
T0*/
_output_shapes
:џџџџџџџџџ `Q
BN6_/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75p

BN6_/add_2AddV2BN6_/moments/Squeeze_1:output:0BN6_/add_2/y:output:0*
T0*
_output_shapes
:`F
	BN6_/SqrtSqrtBN6_/add_2:z:0*
T0*
_output_shapes
:`n
BN6_/truedivRealDivBN6_/sub:z:0BN6_/Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ `p
BN6_/ReadVariableOp_4ReadVariableOpbn6__readvariableop_4_resource*
_output_shapes
:`*
dtype0|

BN6_/mul_4MulBN6_/ReadVariableOp_4:value:0BN6_/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `x
BN6_/add_3/ReadVariableOpReadVariableOp"bn6__add_3_readvariableop_resource*
_output_shapes
:`*
dtype0

BN6_/add_3AddV2BN6_/mul_4:z:0!BN6_/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `^
re_lu_5/ReluReluBN6_/add_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `Ж
average_pooling2d/AvgPoolAvgPoolre_lu_5/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ`*
ksize
 *
paddingVALID*
strides
f
FC1_preFlatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ`   
FC1_preFlatten1/ReshapeReshape"average_pooling2d/AvgPool:output:0FC1_preFlatten1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`~
FC1_/MatMul/ReadVariableOpReadVariableOp#fc1__matmul_readvariableop_resource*
_output_shapes

:`*
dtype0
FC1_/MatMulMatMul FC1_preFlatten1/Reshape:output:0"FC1_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
FC1_/BiasAdd/ReadVariableOpReadVariableOp$fc1__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
FC1_/BiasAddBiasAddFC1_/MatMul:product:0#FC1_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџc
softmax/SoftmaxSoftmaxFC1_/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten/ReshapeReshapesoftmax/Softmax:softmax:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџП
NoOpNoOp^BN1_/AssignVariableOp^BN1_/AssignVariableOp_1^BN1_/AssignVariableOp_2^BN1_/AssignVariableOp_3^BN1_/ReadVariableOp^BN1_/ReadVariableOp_1^BN1_/ReadVariableOp_2^BN1_/ReadVariableOp_3^BN1_/ReadVariableOp_4^BN1_/add_3/ReadVariableOp^BN2_/AssignVariableOp^BN2_/AssignVariableOp_1^BN2_/AssignVariableOp_2^BN2_/AssignVariableOp_3^BN2_/ReadVariableOp^BN2_/ReadVariableOp_1^BN2_/ReadVariableOp_2^BN2_/ReadVariableOp_3^BN2_/ReadVariableOp_4^BN2_/add_3/ReadVariableOp^BN3_/AssignVariableOp^BN3_/AssignVariableOp_1^BN3_/AssignVariableOp_2^BN3_/AssignVariableOp_3^BN3_/ReadVariableOp^BN3_/ReadVariableOp_1^BN3_/ReadVariableOp_2^BN3_/ReadVariableOp_3^BN3_/ReadVariableOp_4^BN3_/add_3/ReadVariableOp^BN4_/AssignVariableOp^BN4_/AssignVariableOp_1^BN4_/AssignVariableOp_2^BN4_/AssignVariableOp_3^BN4_/ReadVariableOp^BN4_/ReadVariableOp_1^BN4_/ReadVariableOp_2^BN4_/ReadVariableOp_3^BN4_/ReadVariableOp_4^BN4_/add_3/ReadVariableOp^BN5_/AssignVariableOp^BN5_/AssignVariableOp_1^BN5_/AssignVariableOp_2^BN5_/AssignVariableOp_3^BN5_/ReadVariableOp^BN5_/ReadVariableOp_1^BN5_/ReadVariableOp_2^BN5_/ReadVariableOp_3^BN5_/ReadVariableOp_4^BN5_/add_3/ReadVariableOp^BN6_/AssignVariableOp^BN6_/AssignVariableOp_1^BN6_/AssignVariableOp_2^BN6_/AssignVariableOp_3^BN6_/ReadVariableOp^BN6_/ReadVariableOp_1^BN6_/ReadVariableOp_2^BN6_/ReadVariableOp_3^BN6_/ReadVariableOp_4^BN6_/add_3/ReadVariableOp^CNN1_/BiasAdd/ReadVariableOp^CNN1_/Conv2D/ReadVariableOp^CNN2_/BiasAdd/ReadVariableOp^CNN2_/Conv2D/ReadVariableOp^CNN3_/BiasAdd/ReadVariableOp^CNN3_/Conv2D/ReadVariableOp^CNN4_/BiasAdd/ReadVariableOp^CNN4_/Conv2D/ReadVariableOp^CNN5_/BiasAdd/ReadVariableOp^CNN5_/Conv2D/ReadVariableOp^CNN6_/BiasAdd/ReadVariableOp^CNN6_/Conv2D/ReadVariableOp^FC1_/BiasAdd/ReadVariableOp^FC1_/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2.
BN1_/AssignVariableOpBN1_/AssignVariableOp22
BN1_/AssignVariableOp_1BN1_/AssignVariableOp_122
BN1_/AssignVariableOp_2BN1_/AssignVariableOp_222
BN1_/AssignVariableOp_3BN1_/AssignVariableOp_32*
BN1_/ReadVariableOpBN1_/ReadVariableOp2.
BN1_/ReadVariableOp_1BN1_/ReadVariableOp_12.
BN1_/ReadVariableOp_2BN1_/ReadVariableOp_22.
BN1_/ReadVariableOp_3BN1_/ReadVariableOp_32.
BN1_/ReadVariableOp_4BN1_/ReadVariableOp_426
BN1_/add_3/ReadVariableOpBN1_/add_3/ReadVariableOp2.
BN2_/AssignVariableOpBN2_/AssignVariableOp22
BN2_/AssignVariableOp_1BN2_/AssignVariableOp_122
BN2_/AssignVariableOp_2BN2_/AssignVariableOp_222
BN2_/AssignVariableOp_3BN2_/AssignVariableOp_32*
BN2_/ReadVariableOpBN2_/ReadVariableOp2.
BN2_/ReadVariableOp_1BN2_/ReadVariableOp_12.
BN2_/ReadVariableOp_2BN2_/ReadVariableOp_22.
BN2_/ReadVariableOp_3BN2_/ReadVariableOp_32.
BN2_/ReadVariableOp_4BN2_/ReadVariableOp_426
BN2_/add_3/ReadVariableOpBN2_/add_3/ReadVariableOp2.
BN3_/AssignVariableOpBN3_/AssignVariableOp22
BN3_/AssignVariableOp_1BN3_/AssignVariableOp_122
BN3_/AssignVariableOp_2BN3_/AssignVariableOp_222
BN3_/AssignVariableOp_3BN3_/AssignVariableOp_32*
BN3_/ReadVariableOpBN3_/ReadVariableOp2.
BN3_/ReadVariableOp_1BN3_/ReadVariableOp_12.
BN3_/ReadVariableOp_2BN3_/ReadVariableOp_22.
BN3_/ReadVariableOp_3BN3_/ReadVariableOp_32.
BN3_/ReadVariableOp_4BN3_/ReadVariableOp_426
BN3_/add_3/ReadVariableOpBN3_/add_3/ReadVariableOp2.
BN4_/AssignVariableOpBN4_/AssignVariableOp22
BN4_/AssignVariableOp_1BN4_/AssignVariableOp_122
BN4_/AssignVariableOp_2BN4_/AssignVariableOp_222
BN4_/AssignVariableOp_3BN4_/AssignVariableOp_32*
BN4_/ReadVariableOpBN4_/ReadVariableOp2.
BN4_/ReadVariableOp_1BN4_/ReadVariableOp_12.
BN4_/ReadVariableOp_2BN4_/ReadVariableOp_22.
BN4_/ReadVariableOp_3BN4_/ReadVariableOp_32.
BN4_/ReadVariableOp_4BN4_/ReadVariableOp_426
BN4_/add_3/ReadVariableOpBN4_/add_3/ReadVariableOp2.
BN5_/AssignVariableOpBN5_/AssignVariableOp22
BN5_/AssignVariableOp_1BN5_/AssignVariableOp_122
BN5_/AssignVariableOp_2BN5_/AssignVariableOp_222
BN5_/AssignVariableOp_3BN5_/AssignVariableOp_32*
BN5_/ReadVariableOpBN5_/ReadVariableOp2.
BN5_/ReadVariableOp_1BN5_/ReadVariableOp_12.
BN5_/ReadVariableOp_2BN5_/ReadVariableOp_22.
BN5_/ReadVariableOp_3BN5_/ReadVariableOp_32.
BN5_/ReadVariableOp_4BN5_/ReadVariableOp_426
BN5_/add_3/ReadVariableOpBN5_/add_3/ReadVariableOp2.
BN6_/AssignVariableOpBN6_/AssignVariableOp22
BN6_/AssignVariableOp_1BN6_/AssignVariableOp_122
BN6_/AssignVariableOp_2BN6_/AssignVariableOp_222
BN6_/AssignVariableOp_3BN6_/AssignVariableOp_32*
BN6_/ReadVariableOpBN6_/ReadVariableOp2.
BN6_/ReadVariableOp_1BN6_/ReadVariableOp_12.
BN6_/ReadVariableOp_2BN6_/ReadVariableOp_22.
BN6_/ReadVariableOp_3BN6_/ReadVariableOp_32.
BN6_/ReadVariableOp_4BN6_/ReadVariableOp_426
BN6_/add_3/ReadVariableOpBN6_/add_3/ReadVariableOp2<
CNN1_/BiasAdd/ReadVariableOpCNN1_/BiasAdd/ReadVariableOp2:
CNN1_/Conv2D/ReadVariableOpCNN1_/Conv2D/ReadVariableOp2<
CNN2_/BiasAdd/ReadVariableOpCNN2_/BiasAdd/ReadVariableOp2:
CNN2_/Conv2D/ReadVariableOpCNN2_/Conv2D/ReadVariableOp2<
CNN3_/BiasAdd/ReadVariableOpCNN3_/BiasAdd/ReadVariableOp2:
CNN3_/Conv2D/ReadVariableOpCNN3_/Conv2D/ReadVariableOp2<
CNN4_/BiasAdd/ReadVariableOpCNN4_/BiasAdd/ReadVariableOp2:
CNN4_/Conv2D/ReadVariableOpCNN4_/Conv2D/ReadVariableOp2<
CNN5_/BiasAdd/ReadVariableOpCNN5_/BiasAdd/ReadVariableOp2:
CNN5_/Conv2D/ReadVariableOpCNN5_/Conv2D/ReadVariableOp2<
CNN6_/BiasAdd/ReadVariableOpCNN6_/BiasAdd/ReadVariableOp2:
CNN6_/Conv2D/ReadVariableOpCNN6_/Conv2D/ReadVariableOp2:
FC1_/BiasAdd/ReadVariableOpFC1_/BiasAdd/ReadVariableOp28
FC1_/MatMul/ReadVariableOpFC1_/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
C
'__inference_re_lu_4_layer_call_fn_17411

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_14746h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_16706
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_16696
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ъ
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_17170

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџ c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ :X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ѕ
П
$__inference_BN5__layer_call_fn_17348

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN5__layer_call_and_return_conditional_losses_15092w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

ж
?__inference_BN5__layer_call_and_return_conditional_losses_14731

inputs)
sub_readvariableop_resource:@%
readvariableop_resource:@'
readvariableop_1_resource:@+
add_1_readvariableop_resource:@
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:@*
dtype0h
subSubinputssub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:@:
SqrtSqrtadd:z:0*
T0*
_output_shapes
:@_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0k
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype0o
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@`
IdentityIdentity	add_1:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

ж
?__inference_BN5__layer_call_and_return_conditional_losses_17367

inputs)
sub_readvariableop_resource:@%
readvariableop_resource:@'
readvariableop_1_resource:@+
add_1_readvariableop_resource:@
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:@*
dtype0h
subSubinputssub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:@:
SqrtSqrtadd:z:0*
T0*
_output_shapes
:@_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0k
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype0o
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@`
IdentityIdentity	add_1:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17426

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

љ
@__inference_CNN2__layer_call_and_return_conditional_losses_16953

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_14640

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџ c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ :X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17303

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16691
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Љ
П
$__inference_BN1__layer_call_fn_16856

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN1__layer_call_and_return_conditional_losses_15420x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ж
?__inference_BN2__layer_call_and_return_conditional_losses_14572

inputs)
sub_readvariableop_resource:%
readvariableop_resource:'
readvariableop_1_resource:+
add_1_readvariableop_resource:
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
::
SqrtSqrtadd:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ

%__inference_CNN1__layer_call_fn_16820

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN1__layer_call_and_return_conditional_losses_14494x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х

%__inference_CNN6__layer_call_fn_17435

inputs!
unknown:@`
	unknown_0:`
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN6__layer_call_and_return_conditional_losses_14759w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ @
 
_user_specified_nameinputs
Ё*
Ъ
?__inference_BN3__layer_call_and_return_conditional_losses_17160

inputs%
readvariableop_resource: '
readvariableop_2_resource: '
readvariableop_4_resource: +
add_3_readvariableop_resource: 
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
: E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
: L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
: I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
: *
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџ L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
: <
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
: `
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
: *
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
: *
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16796
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:D @

_output_shapes
:`
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
*
Ъ
?__inference_BN5__layer_call_and_return_conditional_losses_17406

inputs%
readvariableop_resource:@'
readvariableop_2_resource:@'
readvariableop_4_resource:@+
add_3_readvariableop_resource:@
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:@E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:@И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:@*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:@*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:@L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:@I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:@Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:@*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(f
subSubinputsmoments/Squeeze:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:@<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:@_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:@*
dtype0m
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:@*
dtype0q
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@`
IdentityIdentity	add_3:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
щ

%__inference_CNN3__layer_call_fn_17066

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN3__layer_call_and_return_conditional_losses_14600x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш
\
@__inference_re_lu_layer_call_and_return_conditional_losses_14534

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_14450

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_17539

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ `:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_14426

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
њl
Ќ
@__inference_model_layer_call_and_return_conditional_losses_15632

inputs%
cnn1__15527:
cnn1__15529:

bn1__15532:

bn1__15534:

bn1__15536:

bn1__15538:%
cnn2__15543:
cnn2__15545:

bn2__15548:

bn2__15550:

bn2__15552:

bn2__15554:%
cnn3__15559: 
cnn3__15561: 

bn3__15564: 

bn3__15566: 

bn3__15568: 

bn3__15570: %
cnn4__15575: 0
cnn4__15577:0

bn4__15580:0

bn4__15582:0

bn4__15584:0

bn4__15586:0%
cnn5__15591:0@
cnn5__15593:@

bn5__15596:@

bn5__15598:@

bn5__15600:@

bn5__15602:@%
cnn6__15607:@`
cnn6__15609:`

bn6__15612:`

bn6__15614:`

bn6__15616:`

bn6__15618:`

fc1__15624:`

fc1__15626:
identityЂBN1_/StatefulPartitionedCallЂBN2_/StatefulPartitionedCallЂBN3_/StatefulPartitionedCallЂBN4_/StatefulPartitionedCallЂBN5_/StatefulPartitionedCallЂBN6_/StatefulPartitionedCallЂCNN1_/StatefulPartitionedCallЂCNN2_/StatefulPartitionedCallЂCNN3_/StatefulPartitionedCallЂCNN4_/StatefulPartitionedCallЂCNN5_/StatefulPartitionedCallЂCNN6_/StatefulPartitionedCallЂFC1_/StatefulPartitionedCallэ
CNN1_/StatefulPartitionedCallStatefulPartitionedCallinputscnn1__15527cnn1__15529*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN1__layer_call_and_return_conditional_losses_14494Ѓ
BN1_/StatefulPartitionedCallStatefulPartitionedCall&CNN1_/StatefulPartitionedCall:output:0
bn1__15532
bn1__15534
bn1__15536
bn1__15538*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN1__layer_call_and_return_conditional_losses_15420м
re_lu/PartitionedCallPartitionedCall%BN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_14534х
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_14414
CNN2_/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0cnn2__15543cnn2__15545*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN2__layer_call_and_return_conditional_losses_14547Ѓ
BN2_/StatefulPartitionedCallStatefulPartitionedCall&CNN2_/StatefulPartitionedCall:output:0
bn2__15548
bn2__15550
bn2__15552
bn2__15554*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN2__layer_call_and_return_conditional_losses_15338р
re_lu_1/PartitionedCallPartitionedCall%BN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_14587ы
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_14426
CNN3_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0cnn3__15559cnn3__15561*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN3__layer_call_and_return_conditional_losses_14600Ѓ
BN3_/StatefulPartitionedCallStatefulPartitionedCall&CNN3_/StatefulPartitionedCall:output:0
bn3__15564
bn3__15566
bn3__15568
bn3__15570*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN3__layer_call_and_return_conditional_losses_15256р
re_lu_2/PartitionedCallPartitionedCall%BN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_14640ы
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_14438
CNN4_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0cnn4__15575cnn4__15577*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN4__layer_call_and_return_conditional_losses_14653Ѓ
BN4_/StatefulPartitionedCallStatefulPartitionedCall&CNN4_/StatefulPartitionedCall:output:0
bn4__15580
bn4__15582
bn4__15584
bn4__15586*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN4__layer_call_and_return_conditional_losses_15174р
re_lu_3/PartitionedCallPartitionedCall%BN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_14693ъ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_14450
CNN5_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0cnn5__15591cnn5__15593*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN5__layer_call_and_return_conditional_losses_14706Ђ
BN5_/StatefulPartitionedCallStatefulPartitionedCall&CNN5_/StatefulPartitionedCall:output:0
bn5__15596
bn5__15598
bn5__15600
bn5__15602*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN5__layer_call_and_return_conditional_losses_15092п
re_lu_4/PartitionedCallPartitionedCall%BN5_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_14746ъ
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14462
CNN6_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0cnn6__15607cnn6__15609*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN6__layer_call_and_return_conditional_losses_14759Ђ
BN6_/StatefulPartitionedCallStatefulPartitionedCall&CNN6_/StatefulPartitionedCall:output:0
bn6__15612
bn6__15614
bn6__15616
bn6__15618*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN6__layer_call_and_return_conditional_losses_15010п
re_lu_5/PartitionedCallPartitionedCall%BN6_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_14799ю
!average_pooling2d/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_14474ь
FC1_preFlatten1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_14808
FC1_/StatefulPartitionedCallStatefulPartitionedCall(FC1_preFlatten1/PartitionedCall:output:0
fc1__15624
fc1__15626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_FC1__layer_call_and_return_conditional_losses_14820з
softmax/PartitionedCallPartitionedCall%FC1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_14831в
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14839o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџп
NoOpNoOp^BN1_/StatefulPartitionedCall^BN2_/StatefulPartitionedCall^BN3_/StatefulPartitionedCall^BN4_/StatefulPartitionedCall^BN5_/StatefulPartitionedCall^BN6_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^CNN5_/StatefulPartitionedCall^CNN6_/StatefulPartitionedCall^FC1_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN1_/StatefulPartitionedCallBN1_/StatefulPartitionedCall2<
BN2_/StatefulPartitionedCallBN2_/StatefulPartitionedCall2<
BN3_/StatefulPartitionedCallBN3_/StatefulPartitionedCall2<
BN4_/StatefulPartitionedCallBN4_/StatefulPartitionedCall2<
BN5_/StatefulPartitionedCallBN5_/StatefulPartitionedCall2<
BN6_/StatefulPartitionedCallBN6_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2>
CNN5_/StatefulPartitionedCallCNN5_/StatefulPartitionedCall2>
CNN6_/StatefulPartitionedCallCNN6_/StatefulPartitionedCall2<
FC1_/StatefulPartitionedCallFC1_/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16801
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:D @

_output_shapes
:`
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_16934

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё*
Ъ
?__inference_BN1__layer_call_and_return_conditional_losses_16914

inputs%
readvariableop_resource:'
readvariableop_2_resource:'
readvariableop_4_resource:+
add_3_readvariableop_resource:
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџL
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ

љ
@__inference_CNN6__layer_call_and_return_conditional_losses_17445

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ @
 
_user_specified_nameinputs
Ё*
Ъ
?__inference_BN4__layer_call_and_return_conditional_losses_15174

inputs%
readvariableop_resource:0'
readvariableop_2_resource:0'
readvariableop_4_resource:0+
add_3_readvariableop_resource:0
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:0*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:0
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:0*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:0*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:0*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:0E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:0И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:0*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:0*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:0L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:0I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:0Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:0*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:0<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:0`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ0f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:0*
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:0*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ0: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
ф
њ
%__inference_model_layer_call_fn_15792
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 0

unknown_18:0

unknown_19:0

unknown_20:0

unknown_21:0

unknown_22:0$

unknown_23:0@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@`

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:
identityЂStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*<
_read_only_resource_inputs
 #$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_15632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Т	
№
?__inference_FC1__layer_call_and_return_conditional_losses_14820

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Ё*
Ъ
?__inference_BN2__layer_call_and_return_conditional_losses_15338

inputs%
readvariableop_resource:'
readvariableop_2_resource:'
readvariableop_4_resource:+
add_3_readvariableop_resource:
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџL
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№
њ
%__inference_model_layer_call_fn_14921
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 0

unknown_18:0

unknown_19:0

unknown_20:0

unknown_21:0

unknown_22:0$

unknown_23:0@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@`

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_14842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ц
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_14746

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16756
gradient
variable:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:0: *
	_noinline(:D @

_output_shapes
:0
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
э
љ
%__inference_model_layer_call_fn_16174

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 0

unknown_18:0

unknown_19:0

unknown_20:0

unknown_21:0

unknown_22:0$

unknown_23:0@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@`

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:
identityЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_14842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
^
B__inference_flatten_layer_call_and_return_conditional_losses_14839

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_14693

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџ0c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ0:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ы
^
B__inference_softmax_layer_call_and_return_conditional_losses_14831

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:џџџџџџџџџY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
*
Ъ
?__inference_BN6__layer_call_and_return_conditional_losses_15010

inputs%
readvariableop_resource:`'
readvariableop_2_resource:`'
readvariableop_4_resource:`+
add_3_readvariableop_resource:`
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:`*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:`
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџ `w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:`*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:`L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:`E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:`И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:`*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:`*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:`L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:`I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:`Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:`*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(f
subSubinputsmoments/Squeeze:output:0*
T0*/
_output_shapes
:џџџџџџџџџ `L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:`<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:`_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ `f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:`*
dtype0m
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:`*
dtype0q
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ ``
IdentityIdentity	add_3:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ `: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_16786
gradient"
variable:@`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@`: *
	_noinline(:P L
&
_output_shapes
:@`
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Н
M
1__inference_average_pooling2d_layer_call_fn_17544

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_14474
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ж
?__inference_BN6__layer_call_and_return_conditional_losses_14784

inputs)
sub_readvariableop_resource:`%
readvariableop_resource:`'
readvariableop_1_resource:`+
add_1_readvariableop_resource:`
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:`*
dtype0h
subSubinputssub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:`:
SqrtSqrtadd:z:0*
T0*
_output_shapes
:`_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ `f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0k
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:`*
dtype0o
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ ``
IdentityIdentity	add_1:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ `: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
ъ
^
B__inference_re_lu_3_layer_call_and_return_conditional_losses_17293

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџ0c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ0:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
щ

%__inference_CNN4__layer_call_fn_17189

inputs!
unknown: 0
	unknown_0:0
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN4__layer_call_and_return_conditional_losses_14653x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
М
A
%__inference_re_lu_layer_call_fn_16919

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_14534i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
є
ћ
__inference__traced_save_17833
file_prefix+
'savev2_cnn1__kernel_read_readvariableop)
%savev2_cnn1__bias_read_readvariableop5
1savev2_bn1__custom_batch_beta_read_readvariableop6
2savev2_bn1__custom_batch_gamma_read_readvariableop<
8savev2_bn1__custom_batch_moving_mean_read_readvariableop@
<savev2_bn1__custom_batch_moving_variance_read_readvariableop+
'savev2_cnn2__kernel_read_readvariableop)
%savev2_cnn2__bias_read_readvariableop5
1savev2_bn2__custom_batch_beta_read_readvariableop6
2savev2_bn2__custom_batch_gamma_read_readvariableop<
8savev2_bn2__custom_batch_moving_mean_read_readvariableop@
<savev2_bn2__custom_batch_moving_variance_read_readvariableop+
'savev2_cnn3__kernel_read_readvariableop)
%savev2_cnn3__bias_read_readvariableop5
1savev2_bn3__custom_batch_beta_read_readvariableop6
2savev2_bn3__custom_batch_gamma_read_readvariableop<
8savev2_bn3__custom_batch_moving_mean_read_readvariableop@
<savev2_bn3__custom_batch_moving_variance_read_readvariableop+
'savev2_cnn4__kernel_read_readvariableop)
%savev2_cnn4__bias_read_readvariableop5
1savev2_bn4__custom_batch_beta_read_readvariableop6
2savev2_bn4__custom_batch_gamma_read_readvariableop<
8savev2_bn4__custom_batch_moving_mean_read_readvariableop@
<savev2_bn4__custom_batch_moving_variance_read_readvariableop+
'savev2_cnn5__kernel_read_readvariableop)
%savev2_cnn5__bias_read_readvariableop5
1savev2_bn5__custom_batch_beta_read_readvariableop6
2savev2_bn5__custom_batch_gamma_read_readvariableop<
8savev2_bn5__custom_batch_moving_mean_read_readvariableop@
<savev2_bn5__custom_batch_moving_variance_read_readvariableop+
'savev2_cnn6__kernel_read_readvariableop)
%savev2_cnn6__bias_read_readvariableop5
1savev2_bn6__custom_batch_beta_read_readvariableop6
2savev2_bn6__custom_batch_gamma_read_readvariableop<
8savev2_bn6__custom_batch_moving_mean_read_readvariableop@
<savev2_bn6__custom_batch_moving_variance_read_readvariableop*
&savev2_fc1__kernel_read_readvariableop(
$savev2_fc1__bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop1
-savev2_sgd_m_cnn1__kernel_read_readvariableop/
+savev2_sgd_m_cnn1__bias_read_readvariableop;
7savev2_sgd_m_bn1__custom_batch_beta_read_readvariableop<
8savev2_sgd_m_bn1__custom_batch_gamma_read_readvariableop1
-savev2_sgd_m_cnn2__kernel_read_readvariableop/
+savev2_sgd_m_cnn2__bias_read_readvariableop;
7savev2_sgd_m_bn2__custom_batch_beta_read_readvariableop<
8savev2_sgd_m_bn2__custom_batch_gamma_read_readvariableop1
-savev2_sgd_m_cnn3__kernel_read_readvariableop/
+savev2_sgd_m_cnn3__bias_read_readvariableop;
7savev2_sgd_m_bn3__custom_batch_beta_read_readvariableop<
8savev2_sgd_m_bn3__custom_batch_gamma_read_readvariableop1
-savev2_sgd_m_cnn4__kernel_read_readvariableop/
+savev2_sgd_m_cnn4__bias_read_readvariableop;
7savev2_sgd_m_bn4__custom_batch_beta_read_readvariableop<
8savev2_sgd_m_bn4__custom_batch_gamma_read_readvariableop1
-savev2_sgd_m_cnn5__kernel_read_readvariableop/
+savev2_sgd_m_cnn5__bias_read_readvariableop;
7savev2_sgd_m_bn5__custom_batch_beta_read_readvariableop<
8savev2_sgd_m_bn5__custom_batch_gamma_read_readvariableop1
-savev2_sgd_m_cnn6__kernel_read_readvariableop/
+savev2_sgd_m_cnn6__bias_read_readvariableop;
7savev2_sgd_m_bn6__custom_batch_beta_read_readvariableop<
8savev2_sgd_m_bn6__custom_batch_gamma_read_readvariableop0
,savev2_sgd_m_fc1__kernel_read_readvariableop.
*savev2_sgd_m_fc1__bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
: Ь!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*ѕ 
valueы Bш GB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-1/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-1/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-3/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-3/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-3/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-3/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-5/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-5/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-5/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-5/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-7/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-7/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-7/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-7/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-9/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-9/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-9/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-9/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-11/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-11/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-11/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-11/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHў
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*Ѓ
valueBGB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_cnn1__kernel_read_readvariableop%savev2_cnn1__bias_read_readvariableop1savev2_bn1__custom_batch_beta_read_readvariableop2savev2_bn1__custom_batch_gamma_read_readvariableop8savev2_bn1__custom_batch_moving_mean_read_readvariableop<savev2_bn1__custom_batch_moving_variance_read_readvariableop'savev2_cnn2__kernel_read_readvariableop%savev2_cnn2__bias_read_readvariableop1savev2_bn2__custom_batch_beta_read_readvariableop2savev2_bn2__custom_batch_gamma_read_readvariableop8savev2_bn2__custom_batch_moving_mean_read_readvariableop<savev2_bn2__custom_batch_moving_variance_read_readvariableop'savev2_cnn3__kernel_read_readvariableop%savev2_cnn3__bias_read_readvariableop1savev2_bn3__custom_batch_beta_read_readvariableop2savev2_bn3__custom_batch_gamma_read_readvariableop8savev2_bn3__custom_batch_moving_mean_read_readvariableop<savev2_bn3__custom_batch_moving_variance_read_readvariableop'savev2_cnn4__kernel_read_readvariableop%savev2_cnn4__bias_read_readvariableop1savev2_bn4__custom_batch_beta_read_readvariableop2savev2_bn4__custom_batch_gamma_read_readvariableop8savev2_bn4__custom_batch_moving_mean_read_readvariableop<savev2_bn4__custom_batch_moving_variance_read_readvariableop'savev2_cnn5__kernel_read_readvariableop%savev2_cnn5__bias_read_readvariableop1savev2_bn5__custom_batch_beta_read_readvariableop2savev2_bn5__custom_batch_gamma_read_readvariableop8savev2_bn5__custom_batch_moving_mean_read_readvariableop<savev2_bn5__custom_batch_moving_variance_read_readvariableop'savev2_cnn6__kernel_read_readvariableop%savev2_cnn6__bias_read_readvariableop1savev2_bn6__custom_batch_beta_read_readvariableop2savev2_bn6__custom_batch_gamma_read_readvariableop8savev2_bn6__custom_batch_moving_mean_read_readvariableop<savev2_bn6__custom_batch_moving_variance_read_readvariableop&savev2_fc1__kernel_read_readvariableop$savev2_fc1__bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop-savev2_sgd_m_cnn1__kernel_read_readvariableop+savev2_sgd_m_cnn1__bias_read_readvariableop7savev2_sgd_m_bn1__custom_batch_beta_read_readvariableop8savev2_sgd_m_bn1__custom_batch_gamma_read_readvariableop-savev2_sgd_m_cnn2__kernel_read_readvariableop+savev2_sgd_m_cnn2__bias_read_readvariableop7savev2_sgd_m_bn2__custom_batch_beta_read_readvariableop8savev2_sgd_m_bn2__custom_batch_gamma_read_readvariableop-savev2_sgd_m_cnn3__kernel_read_readvariableop+savev2_sgd_m_cnn3__bias_read_readvariableop7savev2_sgd_m_bn3__custom_batch_beta_read_readvariableop8savev2_sgd_m_bn3__custom_batch_gamma_read_readvariableop-savev2_sgd_m_cnn4__kernel_read_readvariableop+savev2_sgd_m_cnn4__bias_read_readvariableop7savev2_sgd_m_bn4__custom_batch_beta_read_readvariableop8savev2_sgd_m_bn4__custom_batch_gamma_read_readvariableop-savev2_sgd_m_cnn5__kernel_read_readvariableop+savev2_sgd_m_cnn5__bias_read_readvariableop7savev2_sgd_m_bn5__custom_batch_beta_read_readvariableop8savev2_sgd_m_bn5__custom_batch_gamma_read_readvariableop-savev2_sgd_m_cnn6__kernel_read_readvariableop+savev2_sgd_m_cnn6__bias_read_readvariableop7savev2_sgd_m_bn6__custom_batch_beta_read_readvariableop8savev2_sgd_m_bn6__custom_batch_gamma_read_readvariableop,savev2_sgd_m_fc1__kernel_read_readvariableop*savev2_sgd_m_fc1__bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *U
dtypesK
I2G	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*Н
_input_shapesЋ
Ј: ::::::::::::: : : : : : : 0:0:0:0:0:0:0@:@:@:@:@:@:@`:`:`:`:`:`:`:: : ::::::::: : : : : 0:0:0:0:0@:@:@:@:@`:`:`:`:`:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@`:  

_output_shapes
:`: !

_output_shapes
:`: "

_output_shapes
:`: #

_output_shapes
:`: $

_output_shapes
:`:$% 

_output_shapes

:`: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :,)(
&
_output_shapes
:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
::,-(
&
_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::,1(
&
_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: :,5(
&
_output_shapes
: 0: 6

_output_shapes
:0: 7

_output_shapes
:0: 8

_output_shapes
:0:,9(
&
_output_shapes
:0@: :

_output_shapes
:@: ;

_output_shapes
:@: <

_output_shapes
:@:,=(
&
_output_shapes
:@`: >

_output_shapes
:`: ?

_output_shapes
:`: @

_output_shapes
:`:$A 

_output_shapes

:`: B

_output_shapes
::C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: 
Ъ
f
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_17560

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ`   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ`:W S
/
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16721
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_17549

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
 *
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

љ
@__inference_CNN3__layer_call_and_return_conditional_losses_14600

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
П
$__inference_BN2__layer_call_fn_16979

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN2__layer_call_and_return_conditional_losses_15338x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§l
­
@__inference_model_layer_call_and_return_conditional_losses_16008
input_1%
cnn1__15903:
cnn1__15905:

bn1__15908:

bn1__15910:

bn1__15912:

bn1__15914:%
cnn2__15919:
cnn2__15921:

bn2__15924:

bn2__15926:

bn2__15928:

bn2__15930:%
cnn3__15935: 
cnn3__15937: 

bn3__15940: 

bn3__15942: 

bn3__15944: 

bn3__15946: %
cnn4__15951: 0
cnn4__15953:0

bn4__15956:0

bn4__15958:0

bn4__15960:0

bn4__15962:0%
cnn5__15967:0@
cnn5__15969:@

bn5__15972:@

bn5__15974:@

bn5__15976:@

bn5__15978:@%
cnn6__15983:@`
cnn6__15985:`

bn6__15988:`

bn6__15990:`

bn6__15992:`

bn6__15994:`

fc1__16000:`

fc1__16002:
identityЂBN1_/StatefulPartitionedCallЂBN2_/StatefulPartitionedCallЂBN3_/StatefulPartitionedCallЂBN4_/StatefulPartitionedCallЂBN5_/StatefulPartitionedCallЂBN6_/StatefulPartitionedCallЂCNN1_/StatefulPartitionedCallЂCNN2_/StatefulPartitionedCallЂCNN3_/StatefulPartitionedCallЂCNN4_/StatefulPartitionedCallЂCNN5_/StatefulPartitionedCallЂCNN6_/StatefulPartitionedCallЂFC1_/StatefulPartitionedCallю
CNN1_/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn1__15903cnn1__15905*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN1__layer_call_and_return_conditional_losses_14494Ѓ
BN1_/StatefulPartitionedCallStatefulPartitionedCall&CNN1_/StatefulPartitionedCall:output:0
bn1__15908
bn1__15910
bn1__15912
bn1__15914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN1__layer_call_and_return_conditional_losses_15420м
re_lu/PartitionedCallPartitionedCall%BN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_14534х
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_14414
CNN2_/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0cnn2__15919cnn2__15921*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN2__layer_call_and_return_conditional_losses_14547Ѓ
BN2_/StatefulPartitionedCallStatefulPartitionedCall&CNN2_/StatefulPartitionedCall:output:0
bn2__15924
bn2__15926
bn2__15928
bn2__15930*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN2__layer_call_and_return_conditional_losses_15338р
re_lu_1/PartitionedCallPartitionedCall%BN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_14587ы
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_14426
CNN3_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0cnn3__15935cnn3__15937*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN3__layer_call_and_return_conditional_losses_14600Ѓ
BN3_/StatefulPartitionedCallStatefulPartitionedCall&CNN3_/StatefulPartitionedCall:output:0
bn3__15940
bn3__15942
bn3__15944
bn3__15946*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN3__layer_call_and_return_conditional_losses_15256р
re_lu_2/PartitionedCallPartitionedCall%BN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_14640ы
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_14438
CNN4_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0cnn4__15951cnn4__15953*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN4__layer_call_and_return_conditional_losses_14653Ѓ
BN4_/StatefulPartitionedCallStatefulPartitionedCall&CNN4_/StatefulPartitionedCall:output:0
bn4__15956
bn4__15958
bn4__15960
bn4__15962*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN4__layer_call_and_return_conditional_losses_15174р
re_lu_3/PartitionedCallPartitionedCall%BN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_14693ъ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_14450
CNN5_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0cnn5__15967cnn5__15969*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN5__layer_call_and_return_conditional_losses_14706Ђ
BN5_/StatefulPartitionedCallStatefulPartitionedCall&CNN5_/StatefulPartitionedCall:output:0
bn5__15972
bn5__15974
bn5__15976
bn5__15978*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN5__layer_call_and_return_conditional_losses_15092п
re_lu_4/PartitionedCallPartitionedCall%BN5_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_14746ъ
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14462
CNN6_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0cnn6__15983cnn6__15985*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN6__layer_call_and_return_conditional_losses_14759Ђ
BN6_/StatefulPartitionedCallStatefulPartitionedCall&CNN6_/StatefulPartitionedCall:output:0
bn6__15988
bn6__15990
bn6__15992
bn6__15994*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN6__layer_call_and_return_conditional_losses_15010п
re_lu_5/PartitionedCallPartitionedCall%BN6_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_14799ю
!average_pooling2d/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_14474ь
FC1_preFlatten1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_14808
FC1_/StatefulPartitionedCallStatefulPartitionedCall(FC1_preFlatten1/PartitionedCall:output:0
fc1__16000
fc1__16002*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_FC1__layer_call_and_return_conditional_losses_14820з
softmax/PartitionedCallPartitionedCall%FC1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_14831в
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14839o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџп
NoOpNoOp^BN1_/StatefulPartitionedCall^BN2_/StatefulPartitionedCall^BN3_/StatefulPartitionedCall^BN4_/StatefulPartitionedCall^BN5_/StatefulPartitionedCall^BN6_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^CNN5_/StatefulPartitionedCall^CNN6_/StatefulPartitionedCall^FC1_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN1_/StatefulPartitionedCallBN1_/StatefulPartitionedCall2<
BN2_/StatefulPartitionedCallBN2_/StatefulPartitionedCall2<
BN3_/StatefulPartitionedCallBN3_/StatefulPartitionedCall2<
BN4_/StatefulPartitionedCallBN4_/StatefulPartitionedCall2<
BN5_/StatefulPartitionedCallBN5_/StatefulPartitionedCall2<
BN6_/StatefulPartitionedCallBN6_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2>
CNN5_/StatefulPartitionedCallCNN5_/StatefulPartitionedCall2>
CNN6_/StatefulPartitionedCallCNN6_/StatefulPartitionedCall2<
FC1_/StatefulPartitionedCallFC1_/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
m
Ќ
@__inference_model_layer_call_and_return_conditional_losses_14842

inputs%
cnn1__14495:
cnn1__14497:

bn1__14520:

bn1__14522:

bn1__14524:

bn1__14526:%
cnn2__14548:
cnn2__14550:

bn2__14573:

bn2__14575:

bn2__14577:

bn2__14579:%
cnn3__14601: 
cnn3__14603: 

bn3__14626: 

bn3__14628: 

bn3__14630: 

bn3__14632: %
cnn4__14654: 0
cnn4__14656:0

bn4__14679:0

bn4__14681:0

bn4__14683:0

bn4__14685:0%
cnn5__14707:0@
cnn5__14709:@

bn5__14732:@

bn5__14734:@

bn5__14736:@

bn5__14738:@%
cnn6__14760:@`
cnn6__14762:`

bn6__14785:`

bn6__14787:`

bn6__14789:`

bn6__14791:`

fc1__14821:`

fc1__14823:
identityЂBN1_/StatefulPartitionedCallЂBN2_/StatefulPartitionedCallЂBN3_/StatefulPartitionedCallЂBN4_/StatefulPartitionedCallЂBN5_/StatefulPartitionedCallЂBN6_/StatefulPartitionedCallЂCNN1_/StatefulPartitionedCallЂCNN2_/StatefulPartitionedCallЂCNN3_/StatefulPartitionedCallЂCNN4_/StatefulPartitionedCallЂCNN5_/StatefulPartitionedCallЂCNN6_/StatefulPartitionedCallЂFC1_/StatefulPartitionedCallэ
CNN1_/StatefulPartitionedCallStatefulPartitionedCallinputscnn1__14495cnn1__14497*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN1__layer_call_and_return_conditional_losses_14494Ѕ
BN1_/StatefulPartitionedCallStatefulPartitionedCall&CNN1_/StatefulPartitionedCall:output:0
bn1__14520
bn1__14522
bn1__14524
bn1__14526*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN1__layer_call_and_return_conditional_losses_14519м
re_lu/PartitionedCallPartitionedCall%BN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_14534х
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_14414
CNN2_/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0cnn2__14548cnn2__14550*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN2__layer_call_and_return_conditional_losses_14547Ѕ
BN2_/StatefulPartitionedCallStatefulPartitionedCall&CNN2_/StatefulPartitionedCall:output:0
bn2__14573
bn2__14575
bn2__14577
bn2__14579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN2__layer_call_and_return_conditional_losses_14572р
re_lu_1/PartitionedCallPartitionedCall%BN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_14587ы
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_14426
CNN3_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0cnn3__14601cnn3__14603*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN3__layer_call_and_return_conditional_losses_14600Ѕ
BN3_/StatefulPartitionedCallStatefulPartitionedCall&CNN3_/StatefulPartitionedCall:output:0
bn3__14626
bn3__14628
bn3__14630
bn3__14632*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN3__layer_call_and_return_conditional_losses_14625р
re_lu_2/PartitionedCallPartitionedCall%BN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_14640ы
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_14438
CNN4_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0cnn4__14654cnn4__14656*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN4__layer_call_and_return_conditional_losses_14653Ѕ
BN4_/StatefulPartitionedCallStatefulPartitionedCall&CNN4_/StatefulPartitionedCall:output:0
bn4__14679
bn4__14681
bn4__14683
bn4__14685*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN4__layer_call_and_return_conditional_losses_14678р
re_lu_3/PartitionedCallPartitionedCall%BN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_14693ъ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_14450
CNN5_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0cnn5__14707cnn5__14709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN5__layer_call_and_return_conditional_losses_14706Є
BN5_/StatefulPartitionedCallStatefulPartitionedCall&CNN5_/StatefulPartitionedCall:output:0
bn5__14732
bn5__14734
bn5__14736
bn5__14738*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN5__layer_call_and_return_conditional_losses_14731п
re_lu_4/PartitionedCallPartitionedCall%BN5_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_14746ъ
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14462
CNN6_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0cnn6__14760cnn6__14762*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN6__layer_call_and_return_conditional_losses_14759Є
BN6_/StatefulPartitionedCallStatefulPartitionedCall&CNN6_/StatefulPartitionedCall:output:0
bn6__14785
bn6__14787
bn6__14789
bn6__14791*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN6__layer_call_and_return_conditional_losses_14784п
re_lu_5/PartitionedCallPartitionedCall%BN6_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_14799ю
!average_pooling2d/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_14474ь
FC1_preFlatten1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_14808
FC1_/StatefulPartitionedCallStatefulPartitionedCall(FC1_preFlatten1/PartitionedCall:output:0
fc1__14821
fc1__14823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_FC1__layer_call_and_return_conditional_losses_14820з
softmax/PartitionedCallPartitionedCall%FC1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_14831в
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14839o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџп
NoOpNoOp^BN1_/StatefulPartitionedCall^BN2_/StatefulPartitionedCall^BN3_/StatefulPartitionedCall^BN4_/StatefulPartitionedCall^BN5_/StatefulPartitionedCall^BN6_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^CNN5_/StatefulPartitionedCall^CNN6_/StatefulPartitionedCall^FC1_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN1_/StatefulPartitionedCallBN1_/StatefulPartitionedCall2<
BN2_/StatefulPartitionedCallBN2_/StatefulPartitionedCall2<
BN3_/StatefulPartitionedCallBN3_/StatefulPartitionedCall2<
BN4_/StatefulPartitionedCallBN4_/StatefulPartitionedCall2<
BN5_/StatefulPartitionedCallBN5_/StatefulPartitionedCall2<
BN6_/StatefulPartitionedCallBN6_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2>
CNN5_/StatefulPartitionedCallCNN5_/StatefulPartitionedCall2>
CNN6_/StatefulPartitionedCallCNN6_/StatefulPartitionedCall2<
FC1_/StatefulPartitionedCallFC1_/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
П
$__inference_BN3__layer_call_fn_17102

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN3__layer_call_and_return_conditional_losses_15256x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
*
Ъ
?__inference_BN5__layer_call_and_return_conditional_losses_15092

inputs%
readvariableop_resource:@'
readvariableop_2_resource:@'
readvariableop_4_resource:@+
add_3_readvariableop_resource:@
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:@E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:@И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:@*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:@*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:@L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:@I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:@Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:@*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(f
subSubinputsmoments/Squeeze:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:@<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:@_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:@*
dtype0m
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:@*
dtype0q
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@`
IdentityIdentity	add_3:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ы
^
B__inference_softmax_layer_call_and_return_conditional_losses_17589

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:џџџџџџџџџY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Р
C
'__inference_re_lu_2_layer_call_fn_17165

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_14640i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ :X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16716
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ѓ

љ
@__inference_CNN5__layer_call_and_return_conditional_losses_14706

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@0
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_5_layer_call_and_return_conditional_losses_14799

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ `:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
ш
\
@__inference_re_lu_layer_call_and_return_conditional_losses_16924

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
ё
@__inference_model_layer_call_and_return_conditional_losses_16408

inputs>
$cnn1__conv2d_readvariableop_resource:3
%cnn1__biasadd_readvariableop_resource:.
 bn1__sub_readvariableop_resource:*
bn1__readvariableop_resource:,
bn1__readvariableop_1_resource:0
"bn1__add_1_readvariableop_resource:>
$cnn2__conv2d_readvariableop_resource:3
%cnn2__biasadd_readvariableop_resource:.
 bn2__sub_readvariableop_resource:*
bn2__readvariableop_resource:,
bn2__readvariableop_1_resource:0
"bn2__add_1_readvariableop_resource:>
$cnn3__conv2d_readvariableop_resource: 3
%cnn3__biasadd_readvariableop_resource: .
 bn3__sub_readvariableop_resource: *
bn3__readvariableop_resource: ,
bn3__readvariableop_1_resource: 0
"bn3__add_1_readvariableop_resource: >
$cnn4__conv2d_readvariableop_resource: 03
%cnn4__biasadd_readvariableop_resource:0.
 bn4__sub_readvariableop_resource:0*
bn4__readvariableop_resource:0,
bn4__readvariableop_1_resource:00
"bn4__add_1_readvariableop_resource:0>
$cnn5__conv2d_readvariableop_resource:0@3
%cnn5__biasadd_readvariableop_resource:@.
 bn5__sub_readvariableop_resource:@*
bn5__readvariableop_resource:@,
bn5__readvariableop_1_resource:@0
"bn5__add_1_readvariableop_resource:@>
$cnn6__conv2d_readvariableop_resource:@`3
%cnn6__biasadd_readvariableop_resource:`.
 bn6__sub_readvariableop_resource:`*
bn6__readvariableop_resource:`,
bn6__readvariableop_1_resource:`0
"bn6__add_1_readvariableop_resource:`5
#fc1__matmul_readvariableop_resource:`2
$fc1__biasadd_readvariableop_resource:
identityЂBN1_/ReadVariableOpЂBN1_/ReadVariableOp_1ЂBN1_/add_1/ReadVariableOpЂBN1_/sub/ReadVariableOpЂBN2_/ReadVariableOpЂBN2_/ReadVariableOp_1ЂBN2_/add_1/ReadVariableOpЂBN2_/sub/ReadVariableOpЂBN3_/ReadVariableOpЂBN3_/ReadVariableOp_1ЂBN3_/add_1/ReadVariableOpЂBN3_/sub/ReadVariableOpЂBN4_/ReadVariableOpЂBN4_/ReadVariableOp_1ЂBN4_/add_1/ReadVariableOpЂBN4_/sub/ReadVariableOpЂBN5_/ReadVariableOpЂBN5_/ReadVariableOp_1ЂBN5_/add_1/ReadVariableOpЂBN5_/sub/ReadVariableOpЂBN6_/ReadVariableOpЂBN6_/ReadVariableOp_1ЂBN6_/add_1/ReadVariableOpЂBN6_/sub/ReadVariableOpЂCNN1_/BiasAdd/ReadVariableOpЂCNN1_/Conv2D/ReadVariableOpЂCNN2_/BiasAdd/ReadVariableOpЂCNN2_/Conv2D/ReadVariableOpЂCNN3_/BiasAdd/ReadVariableOpЂCNN3_/Conv2D/ReadVariableOpЂCNN4_/BiasAdd/ReadVariableOpЂCNN4_/Conv2D/ReadVariableOpЂCNN5_/BiasAdd/ReadVariableOpЂCNN5_/Conv2D/ReadVariableOpЂCNN6_/BiasAdd/ReadVariableOpЂCNN6_/Conv2D/ReadVariableOpЂFC1_/BiasAdd/ReadVariableOpЂFC1_/MatMul/ReadVariableOp
CNN1_/Conv2D/ReadVariableOpReadVariableOp$cnn1__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0І
CNN1_/Conv2DConv2Dinputs#CNN1_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
~
CNN1_/BiasAdd/ReadVariableOpReadVariableOp%cnn1__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
CNN1_/BiasAddBiasAddCNN1_/Conv2D:output:0$CNN1_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџt
BN1_/sub/ReadVariableOpReadVariableOp bn1__sub_readvariableop_resource*
_output_shapes
:*
dtype0
BN1_/subSubCNN1_/BiasAdd:output:0BN1_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџl
BN1_/ReadVariableOpReadVariableOpbn1__readvariableop_resource*
_output_shapes
:*
dtype0O

BN1_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75h
BN1_/addAddV2BN1_/ReadVariableOp:value:0BN1_/add/y:output:0*
T0*
_output_shapes
:D
	BN1_/SqrtSqrtBN1_/add:z:0*
T0*
_output_shapes
:o
BN1_/truedivRealDivBN1_/sub:z:0BN1_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџp
BN1_/ReadVariableOp_1ReadVariableOpbn1__readvariableop_1_resource*
_output_shapes
:*
dtype0{
BN1_/mulMulBN1_/ReadVariableOp_1:value:0BN1_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџx
BN1_/add_1/ReadVariableOpReadVariableOp"bn1__add_1_readvariableop_resource*
_output_shapes
:*
dtype0

BN1_/add_1AddV2BN1_/mul:z:0!BN1_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ]

re_lu/ReluReluBN1_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџЈ
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

CNN2_/Conv2D/ReadVariableOpReadVariableOp$cnn2__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0О
CNN2_/Conv2DConv2Dmax_pooling2d/MaxPool:output:0#CNN2_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
~
CNN2_/BiasAdd/ReadVariableOpReadVariableOp%cnn2__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
CNN2_/BiasAddBiasAddCNN2_/Conv2D:output:0$CNN2_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџt
BN2_/sub/ReadVariableOpReadVariableOp bn2__sub_readvariableop_resource*
_output_shapes
:*
dtype0
BN2_/subSubCNN2_/BiasAdd:output:0BN2_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџl
BN2_/ReadVariableOpReadVariableOpbn2__readvariableop_resource*
_output_shapes
:*
dtype0O

BN2_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75h
BN2_/addAddV2BN2_/ReadVariableOp:value:0BN2_/add/y:output:0*
T0*
_output_shapes
:D
	BN2_/SqrtSqrtBN2_/add:z:0*
T0*
_output_shapes
:o
BN2_/truedivRealDivBN2_/sub:z:0BN2_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџp
BN2_/ReadVariableOp_1ReadVariableOpbn2__readvariableop_1_resource*
_output_shapes
:*
dtype0{
BN2_/mulMulBN2_/ReadVariableOp_1:value:0BN2_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџx
BN2_/add_1/ReadVariableOpReadVariableOp"bn2__add_1_readvariableop_resource*
_output_shapes
:*
dtype0

BN2_/add_1AddV2BN2_/mul:z:0!BN2_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ_
re_lu_1/ReluReluBN2_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџЌ
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

CNN3_/Conv2D/ReadVariableOpReadVariableOp$cnn3__conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Р
CNN3_/Conv2DConv2D max_pooling2d_1/MaxPool:output:0#CNN3_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
~
CNN3_/BiasAdd/ReadVariableOpReadVariableOp%cnn3__biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
CNN3_/BiasAddBiasAddCNN3_/Conv2D:output:0$CNN3_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ t
BN3_/sub/ReadVariableOpReadVariableOp bn3__sub_readvariableop_resource*
_output_shapes
: *
dtype0
BN3_/subSubCNN3_/BiasAdd:output:0BN3_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ l
BN3_/ReadVariableOpReadVariableOpbn3__readvariableop_resource*
_output_shapes
: *
dtype0O

BN3_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75h
BN3_/addAddV2BN3_/ReadVariableOp:value:0BN3_/add/y:output:0*
T0*
_output_shapes
: D
	BN3_/SqrtSqrtBN3_/add:z:0*
T0*
_output_shapes
: o
BN3_/truedivRealDivBN3_/sub:z:0BN3_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ p
BN3_/ReadVariableOp_1ReadVariableOpbn3__readvariableop_1_resource*
_output_shapes
: *
dtype0{
BN3_/mulMulBN3_/ReadVariableOp_1:value:0BN3_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ x
BN3_/add_1/ReadVariableOpReadVariableOp"bn3__add_1_readvariableop_resource*
_output_shapes
: *
dtype0

BN3_/add_1AddV2BN3_/mul:z:0!BN3_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ _
re_lu_2/ReluReluBN3_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ Ќ
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides

CNN4_/Conv2D/ReadVariableOpReadVariableOp$cnn4__conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Р
CNN4_/Conv2DConv2D max_pooling2d_2/MaxPool:output:0#CNN4_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides
~
CNN4_/BiasAdd/ReadVariableOpReadVariableOp%cnn4__biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
CNN4_/BiasAddBiasAddCNN4_/Conv2D:output:0$CNN4_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0t
BN4_/sub/ReadVariableOpReadVariableOp bn4__sub_readvariableop_resource*
_output_shapes
:0*
dtype0
BN4_/subSubCNN4_/BiasAdd:output:0BN4_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0l
BN4_/ReadVariableOpReadVariableOpbn4__readvariableop_resource*
_output_shapes
:0*
dtype0O

BN4_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75h
BN4_/addAddV2BN4_/ReadVariableOp:value:0BN4_/add/y:output:0*
T0*
_output_shapes
:0D
	BN4_/SqrtSqrtBN4_/add:z:0*
T0*
_output_shapes
:0o
BN4_/truedivRealDivBN4_/sub:z:0BN4_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ0p
BN4_/ReadVariableOp_1ReadVariableOpbn4__readvariableop_1_resource*
_output_shapes
:0*
dtype0{
BN4_/mulMulBN4_/ReadVariableOp_1:value:0BN4_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0x
BN4_/add_1/ReadVariableOpReadVariableOp"bn4__add_1_readvariableop_resource*
_output_shapes
:0*
dtype0

BN4_/add_1AddV2BN4_/mul:z:0!BN4_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0_
re_lu_3/ReluReluBN4_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0Ћ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@0*
ksize
*
paddingVALID*
strides

CNN5_/Conv2D/ReadVariableOpReadVariableOp$cnn5__conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0П
CNN5_/Conv2DConv2D max_pooling2d_3/MaxPool:output:0#CNN5_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
~
CNN5_/BiasAdd/ReadVariableOpReadVariableOp%cnn5__biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
CNN5_/BiasAddBiasAddCNN5_/Conv2D:output:0$CNN5_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@t
BN5_/sub/ReadVariableOpReadVariableOp bn5__sub_readvariableop_resource*
_output_shapes
:@*
dtype0
BN5_/subSubCNN5_/BiasAdd:output:0BN5_/sub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@l
BN5_/ReadVariableOpReadVariableOpbn5__readvariableop_resource*
_output_shapes
:@*
dtype0O

BN5_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75h
BN5_/addAddV2BN5_/ReadVariableOp:value:0BN5_/add/y:output:0*
T0*
_output_shapes
:@D
	BN5_/SqrtSqrtBN5_/add:z:0*
T0*
_output_shapes
:@n
BN5_/truedivRealDivBN5_/sub:z:0BN5_/Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@p
BN5_/ReadVariableOp_1ReadVariableOpbn5__readvariableop_1_resource*
_output_shapes
:@*
dtype0z
BN5_/mulMulBN5_/ReadVariableOp_1:value:0BN5_/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@x
BN5_/add_1/ReadVariableOpReadVariableOp"bn5__add_1_readvariableop_resource*
_output_shapes
:@*
dtype0~

BN5_/add_1AddV2BN5_/mul:z:0!BN5_/add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@^
re_lu_4/ReluReluBN5_/add_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@Ћ
max_pooling2d_4/MaxPoolMaxPoolre_lu_4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides

CNN6_/Conv2D/ReadVariableOpReadVariableOp$cnn6__conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0П
CNN6_/Conv2DConv2D max_pooling2d_4/MaxPool:output:0#CNN6_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `*
paddingSAME*
strides
~
CNN6_/BiasAdd/ReadVariableOpReadVariableOp%cnn6__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
CNN6_/BiasAddBiasAddCNN6_/Conv2D:output:0$CNN6_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `t
BN6_/sub/ReadVariableOpReadVariableOp bn6__sub_readvariableop_resource*
_output_shapes
:`*
dtype0
BN6_/subSubCNN6_/BiasAdd:output:0BN6_/sub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `l
BN6_/ReadVariableOpReadVariableOpbn6__readvariableop_resource*
_output_shapes
:`*
dtype0O

BN6_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75h
BN6_/addAddV2BN6_/ReadVariableOp:value:0BN6_/add/y:output:0*
T0*
_output_shapes
:`D
	BN6_/SqrtSqrtBN6_/add:z:0*
T0*
_output_shapes
:`n
BN6_/truedivRealDivBN6_/sub:z:0BN6_/Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ `p
BN6_/ReadVariableOp_1ReadVariableOpbn6__readvariableop_1_resource*
_output_shapes
:`*
dtype0z
BN6_/mulMulBN6_/ReadVariableOp_1:value:0BN6_/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `x
BN6_/add_1/ReadVariableOpReadVariableOp"bn6__add_1_readvariableop_resource*
_output_shapes
:`*
dtype0~

BN6_/add_1AddV2BN6_/mul:z:0!BN6_/add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `^
re_lu_5/ReluReluBN6_/add_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `Ж
average_pooling2d/AvgPoolAvgPoolre_lu_5/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ`*
ksize
 *
paddingVALID*
strides
f
FC1_preFlatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ`   
FC1_preFlatten1/ReshapeReshape"average_pooling2d/AvgPool:output:0FC1_preFlatten1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`~
FC1_/MatMul/ReadVariableOpReadVariableOp#fc1__matmul_readvariableop_resource*
_output_shapes

:`*
dtype0
FC1_/MatMulMatMul FC1_preFlatten1/Reshape:output:0"FC1_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|
FC1_/BiasAdd/ReadVariableOpReadVariableOp$fc1__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
FC1_/BiasAddBiasAddFC1_/MatMul:product:0#FC1_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџc
softmax/SoftmaxSoftmaxFC1_/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten/ReshapeReshapesoftmax/Softmax:softmax:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџg
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЧ
NoOpNoOp^BN1_/ReadVariableOp^BN1_/ReadVariableOp_1^BN1_/add_1/ReadVariableOp^BN1_/sub/ReadVariableOp^BN2_/ReadVariableOp^BN2_/ReadVariableOp_1^BN2_/add_1/ReadVariableOp^BN2_/sub/ReadVariableOp^BN3_/ReadVariableOp^BN3_/ReadVariableOp_1^BN3_/add_1/ReadVariableOp^BN3_/sub/ReadVariableOp^BN4_/ReadVariableOp^BN4_/ReadVariableOp_1^BN4_/add_1/ReadVariableOp^BN4_/sub/ReadVariableOp^BN5_/ReadVariableOp^BN5_/ReadVariableOp_1^BN5_/add_1/ReadVariableOp^BN5_/sub/ReadVariableOp^BN6_/ReadVariableOp^BN6_/ReadVariableOp_1^BN6_/add_1/ReadVariableOp^BN6_/sub/ReadVariableOp^CNN1_/BiasAdd/ReadVariableOp^CNN1_/Conv2D/ReadVariableOp^CNN2_/BiasAdd/ReadVariableOp^CNN2_/Conv2D/ReadVariableOp^CNN3_/BiasAdd/ReadVariableOp^CNN3_/Conv2D/ReadVariableOp^CNN4_/BiasAdd/ReadVariableOp^CNN4_/Conv2D/ReadVariableOp^CNN5_/BiasAdd/ReadVariableOp^CNN5_/Conv2D/ReadVariableOp^CNN6_/BiasAdd/ReadVariableOp^CNN6_/Conv2D/ReadVariableOp^FC1_/BiasAdd/ReadVariableOp^FC1_/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
BN1_/ReadVariableOpBN1_/ReadVariableOp2.
BN1_/ReadVariableOp_1BN1_/ReadVariableOp_126
BN1_/add_1/ReadVariableOpBN1_/add_1/ReadVariableOp22
BN1_/sub/ReadVariableOpBN1_/sub/ReadVariableOp2*
BN2_/ReadVariableOpBN2_/ReadVariableOp2.
BN2_/ReadVariableOp_1BN2_/ReadVariableOp_126
BN2_/add_1/ReadVariableOpBN2_/add_1/ReadVariableOp22
BN2_/sub/ReadVariableOpBN2_/sub/ReadVariableOp2*
BN3_/ReadVariableOpBN3_/ReadVariableOp2.
BN3_/ReadVariableOp_1BN3_/ReadVariableOp_126
BN3_/add_1/ReadVariableOpBN3_/add_1/ReadVariableOp22
BN3_/sub/ReadVariableOpBN3_/sub/ReadVariableOp2*
BN4_/ReadVariableOpBN4_/ReadVariableOp2.
BN4_/ReadVariableOp_1BN4_/ReadVariableOp_126
BN4_/add_1/ReadVariableOpBN4_/add_1/ReadVariableOp22
BN4_/sub/ReadVariableOpBN4_/sub/ReadVariableOp2*
BN5_/ReadVariableOpBN5_/ReadVariableOp2.
BN5_/ReadVariableOp_1BN5_/ReadVariableOp_126
BN5_/add_1/ReadVariableOpBN5_/add_1/ReadVariableOp22
BN5_/sub/ReadVariableOpBN5_/sub/ReadVariableOp2*
BN6_/ReadVariableOpBN6_/ReadVariableOp2.
BN6_/ReadVariableOp_1BN6_/ReadVariableOp_126
BN6_/add_1/ReadVariableOpBN6_/add_1/ReadVariableOp22
BN6_/sub/ReadVariableOpBN6_/sub/ReadVariableOp2<
CNN1_/BiasAdd/ReadVariableOpCNN1_/BiasAdd/ReadVariableOp2:
CNN1_/Conv2D/ReadVariableOpCNN1_/Conv2D/ReadVariableOp2<
CNN2_/BiasAdd/ReadVariableOpCNN2_/BiasAdd/ReadVariableOp2:
CNN2_/Conv2D/ReadVariableOpCNN2_/Conv2D/ReadVariableOp2<
CNN3_/BiasAdd/ReadVariableOpCNN3_/BiasAdd/ReadVariableOp2:
CNN3_/Conv2D/ReadVariableOpCNN3_/Conv2D/ReadVariableOp2<
CNN4_/BiasAdd/ReadVariableOpCNN4_/BiasAdd/ReadVariableOp2:
CNN4_/Conv2D/ReadVariableOpCNN4_/Conv2D/ReadVariableOp2<
CNN5_/BiasAdd/ReadVariableOpCNN5_/BiasAdd/ReadVariableOp2:
CNN5_/Conv2D/ReadVariableOpCNN5_/Conv2D/ReadVariableOp2<
CNN6_/BiasAdd/ReadVariableOpCNN6_/BiasAdd/ReadVariableOp2:
CNN6_/Conv2D/ReadVariableOpCNN6_/Conv2D/ReadVariableOp2:
FC1_/BiasAdd/ReadVariableOpFC1_/BiasAdd/ReadVariableOp28
FC1_/MatMul/ReadVariableOpFC1_/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
рГ

 __inference__wrapped_model_14405
input_1D
*model_cnn1__conv2d_readvariableop_resource:9
+model_cnn1__biasadd_readvariableop_resource:4
&model_bn1__sub_readvariableop_resource:0
"model_bn1__readvariableop_resource:2
$model_bn1__readvariableop_1_resource:6
(model_bn1__add_1_readvariableop_resource:D
*model_cnn2__conv2d_readvariableop_resource:9
+model_cnn2__biasadd_readvariableop_resource:4
&model_bn2__sub_readvariableop_resource:0
"model_bn2__readvariableop_resource:2
$model_bn2__readvariableop_1_resource:6
(model_bn2__add_1_readvariableop_resource:D
*model_cnn3__conv2d_readvariableop_resource: 9
+model_cnn3__biasadd_readvariableop_resource: 4
&model_bn3__sub_readvariableop_resource: 0
"model_bn3__readvariableop_resource: 2
$model_bn3__readvariableop_1_resource: 6
(model_bn3__add_1_readvariableop_resource: D
*model_cnn4__conv2d_readvariableop_resource: 09
+model_cnn4__biasadd_readvariableop_resource:04
&model_bn4__sub_readvariableop_resource:00
"model_bn4__readvariableop_resource:02
$model_bn4__readvariableop_1_resource:06
(model_bn4__add_1_readvariableop_resource:0D
*model_cnn5__conv2d_readvariableop_resource:0@9
+model_cnn5__biasadd_readvariableop_resource:@4
&model_bn5__sub_readvariableop_resource:@0
"model_bn5__readvariableop_resource:@2
$model_bn5__readvariableop_1_resource:@6
(model_bn5__add_1_readvariableop_resource:@D
*model_cnn6__conv2d_readvariableop_resource:@`9
+model_cnn6__biasadd_readvariableop_resource:`4
&model_bn6__sub_readvariableop_resource:`0
"model_bn6__readvariableop_resource:`2
$model_bn6__readvariableop_1_resource:`6
(model_bn6__add_1_readvariableop_resource:`;
)model_fc1__matmul_readvariableop_resource:`8
*model_fc1__biasadd_readvariableop_resource:
identityЂmodel/BN1_/ReadVariableOpЂmodel/BN1_/ReadVariableOp_1Ђmodel/BN1_/add_1/ReadVariableOpЂmodel/BN1_/sub/ReadVariableOpЂmodel/BN2_/ReadVariableOpЂmodel/BN2_/ReadVariableOp_1Ђmodel/BN2_/add_1/ReadVariableOpЂmodel/BN2_/sub/ReadVariableOpЂmodel/BN3_/ReadVariableOpЂmodel/BN3_/ReadVariableOp_1Ђmodel/BN3_/add_1/ReadVariableOpЂmodel/BN3_/sub/ReadVariableOpЂmodel/BN4_/ReadVariableOpЂmodel/BN4_/ReadVariableOp_1Ђmodel/BN4_/add_1/ReadVariableOpЂmodel/BN4_/sub/ReadVariableOpЂmodel/BN5_/ReadVariableOpЂmodel/BN5_/ReadVariableOp_1Ђmodel/BN5_/add_1/ReadVariableOpЂmodel/BN5_/sub/ReadVariableOpЂmodel/BN6_/ReadVariableOpЂmodel/BN6_/ReadVariableOp_1Ђmodel/BN6_/add_1/ReadVariableOpЂmodel/BN6_/sub/ReadVariableOpЂ"model/CNN1_/BiasAdd/ReadVariableOpЂ!model/CNN1_/Conv2D/ReadVariableOpЂ"model/CNN2_/BiasAdd/ReadVariableOpЂ!model/CNN2_/Conv2D/ReadVariableOpЂ"model/CNN3_/BiasAdd/ReadVariableOpЂ!model/CNN3_/Conv2D/ReadVariableOpЂ"model/CNN4_/BiasAdd/ReadVariableOpЂ!model/CNN4_/Conv2D/ReadVariableOpЂ"model/CNN5_/BiasAdd/ReadVariableOpЂ!model/CNN5_/Conv2D/ReadVariableOpЂ"model/CNN6_/BiasAdd/ReadVariableOpЂ!model/CNN6_/Conv2D/ReadVariableOpЂ!model/FC1_/BiasAdd/ReadVariableOpЂ model/FC1_/MatMul/ReadVariableOp
!model/CNN1_/Conv2D/ReadVariableOpReadVariableOp*model_cnn1__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Г
model/CNN1_/Conv2DConv2Dinput_1)model/CNN1_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

"model/CNN1_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn1__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
model/CNN1_/BiasAddBiasAddmodel/CNN1_/Conv2D:output:0*model/CNN1_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/BN1_/sub/ReadVariableOpReadVariableOp&model_bn1__sub_readvariableop_resource*
_output_shapes
:*
dtype0
model/BN1_/subSubmodel/CNN1_/BiasAdd:output:0%model/BN1_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџx
model/BN1_/ReadVariableOpReadVariableOp"model_bn1__readvariableop_resource*
_output_shapes
:*
dtype0U
model/BN1_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75z
model/BN1_/addAddV2!model/BN1_/ReadVariableOp:value:0model/BN1_/add/y:output:0*
T0*
_output_shapes
:P
model/BN1_/SqrtSqrtmodel/BN1_/add:z:0*
T0*
_output_shapes
:
model/BN1_/truedivRealDivmodel/BN1_/sub:z:0model/BN1_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ|
model/BN1_/ReadVariableOp_1ReadVariableOp$model_bn1__readvariableop_1_resource*
_output_shapes
:*
dtype0
model/BN1_/mulMul#model/BN1_/ReadVariableOp_1:value:0model/BN1_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/BN1_/add_1/ReadVariableOpReadVariableOp(model_bn1__add_1_readvariableop_resource*
_output_shapes
:*
dtype0
model/BN1_/add_1AddV2model/BN1_/mul:z:0'model/BN1_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџi
model/re_lu/ReluRelumodel/BN1_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџД
model/max_pooling2d/MaxPoolMaxPoolmodel/re_lu/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

!model/CNN2_/Conv2D/ReadVariableOpReadVariableOp*model_cnn2__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0а
model/CNN2_/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0)model/CNN2_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

"model/CNN2_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn2__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
model/CNN2_/BiasAddBiasAddmodel/CNN2_/Conv2D:output:0*model/CNN2_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/BN2_/sub/ReadVariableOpReadVariableOp&model_bn2__sub_readvariableop_resource*
_output_shapes
:*
dtype0
model/BN2_/subSubmodel/CNN2_/BiasAdd:output:0%model/BN2_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџx
model/BN2_/ReadVariableOpReadVariableOp"model_bn2__readvariableop_resource*
_output_shapes
:*
dtype0U
model/BN2_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75z
model/BN2_/addAddV2!model/BN2_/ReadVariableOp:value:0model/BN2_/add/y:output:0*
T0*
_output_shapes
:P
model/BN2_/SqrtSqrtmodel/BN2_/add:z:0*
T0*
_output_shapes
:
model/BN2_/truedivRealDivmodel/BN2_/sub:z:0model/BN2_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ|
model/BN2_/ReadVariableOp_1ReadVariableOp$model_bn2__readvariableop_1_resource*
_output_shapes
:*
dtype0
model/BN2_/mulMul#model/BN2_/ReadVariableOp_1:value:0model/BN2_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ
model/BN2_/add_1/ReadVariableOpReadVariableOp(model_bn2__add_1_readvariableop_resource*
_output_shapes
:*
dtype0
model/BN2_/add_1AddV2model/BN2_/mul:z:0'model/BN2_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџk
model/re_lu_1/ReluRelumodel/BN2_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџИ
model/max_pooling2d_1/MaxPoolMaxPool model/re_lu_1/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

!model/CNN3_/Conv2D/ReadVariableOpReadVariableOp*model_cnn3__conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0в
model/CNN3_/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0)model/CNN3_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

"model/CNN3_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn3__biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ђ
model/CNN3_/BiasAddBiasAddmodel/CNN3_/Conv2D:output:0*model/CNN3_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ 
model/BN3_/sub/ReadVariableOpReadVariableOp&model_bn3__sub_readvariableop_resource*
_output_shapes
: *
dtype0
model/BN3_/subSubmodel/CNN3_/BiasAdd:output:0%model/BN3_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ x
model/BN3_/ReadVariableOpReadVariableOp"model_bn3__readvariableop_resource*
_output_shapes
: *
dtype0U
model/BN3_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75z
model/BN3_/addAddV2!model/BN3_/ReadVariableOp:value:0model/BN3_/add/y:output:0*
T0*
_output_shapes
: P
model/BN3_/SqrtSqrtmodel/BN3_/add:z:0*
T0*
_output_shapes
: 
model/BN3_/truedivRealDivmodel/BN3_/sub:z:0model/BN3_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ |
model/BN3_/ReadVariableOp_1ReadVariableOp$model_bn3__readvariableop_1_resource*
_output_shapes
: *
dtype0
model/BN3_/mulMul#model/BN3_/ReadVariableOp_1:value:0model/BN3_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ 
model/BN3_/add_1/ReadVariableOpReadVariableOp(model_bn3__add_1_readvariableop_resource*
_output_shapes
: *
dtype0
model/BN3_/add_1AddV2model/BN3_/mul:z:0'model/BN3_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ k
model/re_lu_2/ReluRelumodel/BN3_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ И
model/max_pooling2d_2/MaxPoolMaxPool model/re_lu_2/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides

!model/CNN4_/Conv2D/ReadVariableOpReadVariableOp*model_cnn4__conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0в
model/CNN4_/Conv2DConv2D&model/max_pooling2d_2/MaxPool:output:0)model/CNN4_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides

"model/CNN4_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn4__biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Ђ
model/CNN4_/BiasAddBiasAddmodel/CNN4_/Conv2D:output:0*model/CNN4_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0
model/BN4_/sub/ReadVariableOpReadVariableOp&model_bn4__sub_readvariableop_resource*
_output_shapes
:0*
dtype0
model/BN4_/subSubmodel/CNN4_/BiasAdd:output:0%model/BN4_/sub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0x
model/BN4_/ReadVariableOpReadVariableOp"model_bn4__readvariableop_resource*
_output_shapes
:0*
dtype0U
model/BN4_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75z
model/BN4_/addAddV2!model/BN4_/ReadVariableOp:value:0model/BN4_/add/y:output:0*
T0*
_output_shapes
:0P
model/BN4_/SqrtSqrtmodel/BN4_/add:z:0*
T0*
_output_shapes
:0
model/BN4_/truedivRealDivmodel/BN4_/sub:z:0model/BN4_/Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ0|
model/BN4_/ReadVariableOp_1ReadVariableOp$model_bn4__readvariableop_1_resource*
_output_shapes
:0*
dtype0
model/BN4_/mulMul#model/BN4_/ReadVariableOp_1:value:0model/BN4_/truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0
model/BN4_/add_1/ReadVariableOpReadVariableOp(model_bn4__add_1_readvariableop_resource*
_output_shapes
:0*
dtype0
model/BN4_/add_1AddV2model/BN4_/mul:z:0'model/BN4_/add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0k
model/re_lu_3/ReluRelumodel/BN4_/add_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0З
model/max_pooling2d_3/MaxPoolMaxPool model/re_lu_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@0*
ksize
*
paddingVALID*
strides

!model/CNN5_/Conv2D/ReadVariableOpReadVariableOp*model_cnn5__conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0б
model/CNN5_/Conv2DConv2D&model/max_pooling2d_3/MaxPool:output:0)model/CNN5_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides

"model/CNN5_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn5__biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ё
model/CNN5_/BiasAddBiasAddmodel/CNN5_/Conv2D:output:0*model/CNN5_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@
model/BN5_/sub/ReadVariableOpReadVariableOp&model_bn5__sub_readvariableop_resource*
_output_shapes
:@*
dtype0
model/BN5_/subSubmodel/CNN5_/BiasAdd:output:0%model/BN5_/sub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@x
model/BN5_/ReadVariableOpReadVariableOp"model_bn5__readvariableop_resource*
_output_shapes
:@*
dtype0U
model/BN5_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75z
model/BN5_/addAddV2!model/BN5_/ReadVariableOp:value:0model/BN5_/add/y:output:0*
T0*
_output_shapes
:@P
model/BN5_/SqrtSqrtmodel/BN5_/add:z:0*
T0*
_output_shapes
:@
model/BN5_/truedivRealDivmodel/BN5_/sub:z:0model/BN5_/Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@@|
model/BN5_/ReadVariableOp_1ReadVariableOp$model_bn5__readvariableop_1_resource*
_output_shapes
:@*
dtype0
model/BN5_/mulMul#model/BN5_/ReadVariableOp_1:value:0model/BN5_/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@
model/BN5_/add_1/ReadVariableOpReadVariableOp(model_bn5__add_1_readvariableop_resource*
_output_shapes
:@*
dtype0
model/BN5_/add_1AddV2model/BN5_/mul:z:0'model/BN5_/add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@j
model/re_lu_4/ReluRelumodel/BN5_/add_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@@З
model/max_pooling2d_4/MaxPoolMaxPool model/re_lu_4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides

!model/CNN6_/Conv2D/ReadVariableOpReadVariableOp*model_cnn6__conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0б
model/CNN6_/Conv2DConv2D&model/max_pooling2d_4/MaxPool:output:0)model/CNN6_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `*
paddingSAME*
strides

"model/CNN6_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn6__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ё
model/CNN6_/BiasAddBiasAddmodel/CNN6_/Conv2D:output:0*model/CNN6_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `
model/BN6_/sub/ReadVariableOpReadVariableOp&model_bn6__sub_readvariableop_resource*
_output_shapes
:`*
dtype0
model/BN6_/subSubmodel/CNN6_/BiasAdd:output:0%model/BN6_/sub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `x
model/BN6_/ReadVariableOpReadVariableOp"model_bn6__readvariableop_resource*
_output_shapes
:`*
dtype0U
model/BN6_/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75z
model/BN6_/addAddV2!model/BN6_/ReadVariableOp:value:0model/BN6_/add/y:output:0*
T0*
_output_shapes
:`P
model/BN6_/SqrtSqrtmodel/BN6_/add:z:0*
T0*
_output_shapes
:`
model/BN6_/truedivRealDivmodel/BN6_/sub:z:0model/BN6_/Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ `|
model/BN6_/ReadVariableOp_1ReadVariableOp$model_bn6__readvariableop_1_resource*
_output_shapes
:`*
dtype0
model/BN6_/mulMul#model/BN6_/ReadVariableOp_1:value:0model/BN6_/truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `
model/BN6_/add_1/ReadVariableOpReadVariableOp(model_bn6__add_1_readvariableop_resource*
_output_shapes
:`*
dtype0
model/BN6_/add_1AddV2model/BN6_/mul:z:0'model/BN6_/add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `j
model/re_lu_5/ReluRelumodel/BN6_/add_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `Т
model/average_pooling2d/AvgPoolAvgPool model/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ`*
ksize
 *
paddingVALID*
strides
l
model/FC1_preFlatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ`   Њ
model/FC1_preFlatten1/ReshapeReshape(model/average_pooling2d/AvgPool:output:0$model/FC1_preFlatten1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
 model/FC1_/MatMul/ReadVariableOpReadVariableOp)model_fc1__matmul_readvariableop_resource*
_output_shapes

:`*
dtype0
model/FC1_/MatMulMatMul&model/FC1_preFlatten1/Reshape:output:0(model/FC1_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!model/FC1_/BiasAdd/ReadVariableOpReadVariableOp*model_fc1__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/FC1_/BiasAddBiasAddmodel/FC1_/MatMul:product:0)model/FC1_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџo
model/softmax/SoftmaxSoftmaxmodel/FC1_/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
model/flatten/ReshapeReshapemodel/softmax/Softmax:softmax:0model/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
IdentityIdentitymodel/flatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЋ

NoOpNoOp^model/BN1_/ReadVariableOp^model/BN1_/ReadVariableOp_1 ^model/BN1_/add_1/ReadVariableOp^model/BN1_/sub/ReadVariableOp^model/BN2_/ReadVariableOp^model/BN2_/ReadVariableOp_1 ^model/BN2_/add_1/ReadVariableOp^model/BN2_/sub/ReadVariableOp^model/BN3_/ReadVariableOp^model/BN3_/ReadVariableOp_1 ^model/BN3_/add_1/ReadVariableOp^model/BN3_/sub/ReadVariableOp^model/BN4_/ReadVariableOp^model/BN4_/ReadVariableOp_1 ^model/BN4_/add_1/ReadVariableOp^model/BN4_/sub/ReadVariableOp^model/BN5_/ReadVariableOp^model/BN5_/ReadVariableOp_1 ^model/BN5_/add_1/ReadVariableOp^model/BN5_/sub/ReadVariableOp^model/BN6_/ReadVariableOp^model/BN6_/ReadVariableOp_1 ^model/BN6_/add_1/ReadVariableOp^model/BN6_/sub/ReadVariableOp#^model/CNN1_/BiasAdd/ReadVariableOp"^model/CNN1_/Conv2D/ReadVariableOp#^model/CNN2_/BiasAdd/ReadVariableOp"^model/CNN2_/Conv2D/ReadVariableOp#^model/CNN3_/BiasAdd/ReadVariableOp"^model/CNN3_/Conv2D/ReadVariableOp#^model/CNN4_/BiasAdd/ReadVariableOp"^model/CNN4_/Conv2D/ReadVariableOp#^model/CNN5_/BiasAdd/ReadVariableOp"^model/CNN5_/Conv2D/ReadVariableOp#^model/CNN6_/BiasAdd/ReadVariableOp"^model/CNN6_/Conv2D/ReadVariableOp"^model/FC1_/BiasAdd/ReadVariableOp!^model/FC1_/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 26
model/BN1_/ReadVariableOpmodel/BN1_/ReadVariableOp2:
model/BN1_/ReadVariableOp_1model/BN1_/ReadVariableOp_12B
model/BN1_/add_1/ReadVariableOpmodel/BN1_/add_1/ReadVariableOp2>
model/BN1_/sub/ReadVariableOpmodel/BN1_/sub/ReadVariableOp26
model/BN2_/ReadVariableOpmodel/BN2_/ReadVariableOp2:
model/BN2_/ReadVariableOp_1model/BN2_/ReadVariableOp_12B
model/BN2_/add_1/ReadVariableOpmodel/BN2_/add_1/ReadVariableOp2>
model/BN2_/sub/ReadVariableOpmodel/BN2_/sub/ReadVariableOp26
model/BN3_/ReadVariableOpmodel/BN3_/ReadVariableOp2:
model/BN3_/ReadVariableOp_1model/BN3_/ReadVariableOp_12B
model/BN3_/add_1/ReadVariableOpmodel/BN3_/add_1/ReadVariableOp2>
model/BN3_/sub/ReadVariableOpmodel/BN3_/sub/ReadVariableOp26
model/BN4_/ReadVariableOpmodel/BN4_/ReadVariableOp2:
model/BN4_/ReadVariableOp_1model/BN4_/ReadVariableOp_12B
model/BN4_/add_1/ReadVariableOpmodel/BN4_/add_1/ReadVariableOp2>
model/BN4_/sub/ReadVariableOpmodel/BN4_/sub/ReadVariableOp26
model/BN5_/ReadVariableOpmodel/BN5_/ReadVariableOp2:
model/BN5_/ReadVariableOp_1model/BN5_/ReadVariableOp_12B
model/BN5_/add_1/ReadVariableOpmodel/BN5_/add_1/ReadVariableOp2>
model/BN5_/sub/ReadVariableOpmodel/BN5_/sub/ReadVariableOp26
model/BN6_/ReadVariableOpmodel/BN6_/ReadVariableOp2:
model/BN6_/ReadVariableOp_1model/BN6_/ReadVariableOp_12B
model/BN6_/add_1/ReadVariableOpmodel/BN6_/add_1/ReadVariableOp2>
model/BN6_/sub/ReadVariableOpmodel/BN6_/sub/ReadVariableOp2H
"model/CNN1_/BiasAdd/ReadVariableOp"model/CNN1_/BiasAdd/ReadVariableOp2F
!model/CNN1_/Conv2D/ReadVariableOp!model/CNN1_/Conv2D/ReadVariableOp2H
"model/CNN2_/BiasAdd/ReadVariableOp"model/CNN2_/BiasAdd/ReadVariableOp2F
!model/CNN2_/Conv2D/ReadVariableOp!model/CNN2_/Conv2D/ReadVariableOp2H
"model/CNN3_/BiasAdd/ReadVariableOp"model/CNN3_/BiasAdd/ReadVariableOp2F
!model/CNN3_/Conv2D/ReadVariableOp!model/CNN3_/Conv2D/ReadVariableOp2H
"model/CNN4_/BiasAdd/ReadVariableOp"model/CNN4_/BiasAdd/ReadVariableOp2F
!model/CNN4_/Conv2D/ReadVariableOp!model/CNN4_/Conv2D/ReadVariableOp2H
"model/CNN5_/BiasAdd/ReadVariableOp"model/CNN5_/BiasAdd/ReadVariableOp2F
!model/CNN5_/Conv2D/ReadVariableOp!model/CNN5_/Conv2D/ReadVariableOp2H
"model/CNN6_/BiasAdd/ReadVariableOp"model/CNN6_/BiasAdd/ReadVariableOp2F
!model/CNN6_/Conv2D/ReadVariableOp!model/CNN6_/Conv2D/ReadVariableOp2F
!model/FC1_/BiasAdd/ReadVariableOp!model/FC1_/BiasAdd/ReadVariableOp2D
 model/FC1_/MatMul/ReadVariableOp model/FC1_/MatMul/ReadVariableOp:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ј

љ
@__inference_CNN3__layer_call_and_return_conditional_losses_17076

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_14587

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј

љ
@__inference_CNN2__layer_call_and_return_conditional_losses_14547

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16791
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:`: *
	_noinline(:D @

_output_shapes
:`
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ё*
Ъ
?__inference_BN4__layer_call_and_return_conditional_losses_17283

inputs%
readvariableop_resource:0'
readvariableop_2_resource:0'
readvariableop_4_resource:0+
add_3_readvariableop_resource:0
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:0*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:0
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:0*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:0*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:0*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:0L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:0E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:0И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:0*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:0*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:0L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:0I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:0Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:0*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:0<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:0`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ0f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:0*
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:0*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ0: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ћ
П
$__inference_BN2__layer_call_fn_16966

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN2__layer_call_and_return_conditional_losses_14572x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ж
?__inference_BN4__layer_call_and_return_conditional_losses_14678

inputs)
sub_readvariableop_resource:0%
readvariableop_resource:0'
readvariableop_1_resource:0+
add_1_readvariableop_resource:0
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:0*
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:0:
SqrtSqrtadd:z:0*
T0*
_output_shapes
:0`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:0*
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0a
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ0: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
х

%__inference_CNN5__layer_call_fn_17312

inputs!
unknown:0@
	unknown_0:@
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN5__layer_call_and_return_conditional_losses_14706w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@0
 
_user_specified_nameinputs

h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_14474

inputs
identityЋ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
 *
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
М
K
/__inference_FC1_preFlatten1_layer_call_fn_17554

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_14808`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ`:W S
/
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Ю
ј
#__inference_signature_wrapper_16093
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: 0

unknown_18:0

unknown_19:0

unknown_20:0

unknown_21:0

unknown_22:0$

unknown_23:0@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29:@`

unknown_30:`

unknown_31:`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_14405o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ћ
J
"__inference__update_step_xla_16731
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
В
^
B__inference_flatten_layer_call_and_return_conditional_losses_17600

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16736
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

ж
?__inference_BN6__layer_call_and_return_conditional_losses_17490

inputs)
sub_readvariableop_resource:`%
readvariableop_resource:`'
readvariableop_1_resource:`+
add_1_readvariableop_resource:`
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:`*
dtype0h
subSubinputssub/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ `b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:`:
SqrtSqrtadd:z:0*
T0*
_output_shapes
:`_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ `f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0k
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:`*
dtype0o
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ ``
IdentityIdentity	add_1:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ `: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_16726
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_16741
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17057

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ц
^
B__inference_re_lu_4_layer_call_and_return_conditional_losses_17416

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:џџџџџџџџџ@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16811
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14462

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
щ

%__inference_CNN2__layer_call_fn_16943

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN2__layer_call_and_return_conditional_losses_14547x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѕ
П
$__inference_BN6__layer_call_fn_17471

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN6__layer_call_and_return_conditional_losses_15010w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ `: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
ъ
^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_17047

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
m
­
@__inference_model_layer_call_and_return_conditional_losses_15900
input_1%
cnn1__15795:
cnn1__15797:

bn1__15800:

bn1__15802:

bn1__15804:

bn1__15806:%
cnn2__15811:
cnn2__15813:

bn2__15816:

bn2__15818:

bn2__15820:

bn2__15822:%
cnn3__15827: 
cnn3__15829: 

bn3__15832: 

bn3__15834: 

bn3__15836: 

bn3__15838: %
cnn4__15843: 0
cnn4__15845:0

bn4__15848:0

bn4__15850:0

bn4__15852:0

bn4__15854:0%
cnn5__15859:0@
cnn5__15861:@

bn5__15864:@

bn5__15866:@

bn5__15868:@

bn5__15870:@%
cnn6__15875:@`
cnn6__15877:`

bn6__15880:`

bn6__15882:`

bn6__15884:`

bn6__15886:`

fc1__15892:`

fc1__15894:
identityЂBN1_/StatefulPartitionedCallЂBN2_/StatefulPartitionedCallЂBN3_/StatefulPartitionedCallЂBN4_/StatefulPartitionedCallЂBN5_/StatefulPartitionedCallЂBN6_/StatefulPartitionedCallЂCNN1_/StatefulPartitionedCallЂCNN2_/StatefulPartitionedCallЂCNN3_/StatefulPartitionedCallЂCNN4_/StatefulPartitionedCallЂCNN5_/StatefulPartitionedCallЂCNN6_/StatefulPartitionedCallЂFC1_/StatefulPartitionedCallю
CNN1_/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn1__15795cnn1__15797*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN1__layer_call_and_return_conditional_losses_14494Ѕ
BN1_/StatefulPartitionedCallStatefulPartitionedCall&CNN1_/StatefulPartitionedCall:output:0
bn1__15800
bn1__15802
bn1__15804
bn1__15806*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN1__layer_call_and_return_conditional_losses_14519м
re_lu/PartitionedCallPartitionedCall%BN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_14534х
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_14414
CNN2_/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0cnn2__15811cnn2__15813*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN2__layer_call_and_return_conditional_losses_14547Ѕ
BN2_/StatefulPartitionedCallStatefulPartitionedCall&CNN2_/StatefulPartitionedCall:output:0
bn2__15816
bn2__15818
bn2__15820
bn2__15822*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN2__layer_call_and_return_conditional_losses_14572р
re_lu_1/PartitionedCallPartitionedCall%BN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_14587ы
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_14426
CNN3_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0cnn3__15827cnn3__15829*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN3__layer_call_and_return_conditional_losses_14600Ѕ
BN3_/StatefulPartitionedCallStatefulPartitionedCall&CNN3_/StatefulPartitionedCall:output:0
bn3__15832
bn3__15834
bn3__15836
bn3__15838*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN3__layer_call_and_return_conditional_losses_14625р
re_lu_2/PartitionedCallPartitionedCall%BN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_14640ы
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_14438
CNN4_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0cnn4__15843cnn4__15845*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN4__layer_call_and_return_conditional_losses_14653Ѕ
BN4_/StatefulPartitionedCallStatefulPartitionedCall&CNN4_/StatefulPartitionedCall:output:0
bn4__15848
bn4__15850
bn4__15852
bn4__15854*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN4__layer_call_and_return_conditional_losses_14678р
re_lu_3/PartitionedCallPartitionedCall%BN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_3_layer_call_and_return_conditional_losses_14693ъ
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_14450
CNN5_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0cnn5__15859cnn5__15861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN5__layer_call_and_return_conditional_losses_14706Є
BN5_/StatefulPartitionedCallStatefulPartitionedCall&CNN5_/StatefulPartitionedCall:output:0
bn5__15864
bn5__15866
bn5__15868
bn5__15870*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN5__layer_call_and_return_conditional_losses_14731п
re_lu_4/PartitionedCallPartitionedCall%BN5_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_4_layer_call_and_return_conditional_losses_14746ъ
max_pooling2d_4/PartitionedCallPartitionedCall re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14462
CNN6_/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0cnn6__15875cnn6__15877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_CNN6__layer_call_and_return_conditional_losses_14759Є
BN6_/StatefulPartitionedCallStatefulPartitionedCall&CNN6_/StatefulPartitionedCall:output:0
bn6__15880
bn6__15882
bn6__15884
bn6__15886*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN6__layer_call_and_return_conditional_losses_14784п
re_lu_5/PartitionedCallPartitionedCall%BN6_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_14799ю
!average_pooling2d/PartitionedCallPartitionedCall re_lu_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_14474ь
FC1_preFlatten1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_14808
FC1_/StatefulPartitionedCallStatefulPartitionedCall(FC1_preFlatten1/PartitionedCall:output:0
fc1__15892
fc1__15894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_FC1__layer_call_and_return_conditional_losses_14820з
softmax/PartitionedCallPartitionedCall%FC1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_14831в
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14839o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџп
NoOpNoOp^BN1_/StatefulPartitionedCall^BN2_/StatefulPartitionedCall^BN3_/StatefulPartitionedCall^BN4_/StatefulPartitionedCall^BN5_/StatefulPartitionedCall^BN6_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^CNN5_/StatefulPartitionedCall^CNN6_/StatefulPartitionedCall^FC1_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN1_/StatefulPartitionedCallBN1_/StatefulPartitionedCall2<
BN2_/StatefulPartitionedCallBN2_/StatefulPartitionedCall2<
BN3_/StatefulPartitionedCallBN3_/StatefulPartitionedCall2<
BN4_/StatefulPartitionedCallBN4_/StatefulPartitionedCall2<
BN5_/StatefulPartitionedCallBN5_/StatefulPartitionedCall2<
BN6_/StatefulPartitionedCallBN6_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2>
CNN5_/StatefulPartitionedCallCNN5_/StatefulPartitionedCall2>
CNN6_/StatefulPartitionedCallCNN6_/StatefulPartitionedCall2<
FC1_/StatefulPartitionedCallFC1_/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
*
Ъ
?__inference_BN6__layer_call_and_return_conditional_losses_17529

inputs%
readvariableop_resource:`'
readvariableop_2_resource:`'
readvariableop_4_resource:`+
add_3_readvariableop_resource:`
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:`*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:`
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*/
_output_shapes
:џџџџџџџџџ `w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:`*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:`*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:`L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:`E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:`И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:`*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:`*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:`L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:`I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:`Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:`*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(f
subSubinputsmoments/Squeeze:output:0*
T0*/
_output_shapes
:џџџџџџџџџ `L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:`<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:`_
truedivRealDivsub:z:0Sqrt:y:0*
T0*/
_output_shapes
:џџџџџџџџџ `f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:`*
dtype0m
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*/
_output_shapes
:џџџџџџџџџ `n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:`*
dtype0q
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ ``
IdentityIdentity	add_3:z:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ `: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16771
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Я
V
"__inference__update_step_xla_16746
gradient"
variable: 0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: 0: *
	_noinline(:P L
&
_output_shapes
: 0
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Л

$__inference_FC1__layer_call_fn_17569

inputs
unknown:`
	unknown_0:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_FC1__layer_call_and_return_conditional_losses_14820o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_14438

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ж
?__inference_BN3__layer_call_and_return_conditional_losses_14625

inputs)
sub_readvariableop_resource: %
readvariableop_resource: '
readvariableop_1_resource: +
add_1_readvariableop_resource: 
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
: :
SqrtSqrtadd:z:0*
T0*
_output_shapes
: `
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ a
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ : : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
М
C
'__inference_re_lu_5_layer_call_fn_17534

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_re_lu_5_layer_call_and_return_conditional_losses_14799h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ `:W S
/
_output_shapes
:џџџџџџџџџ `
 
_user_specified_nameinputs
Ј

љ
@__inference_CNN1__layer_call_and_return_conditional_losses_16830

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16781
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
П
$__inference_BN4__layer_call_fn_17212

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN4__layer_call_and_return_conditional_losses_14678x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs

C
'__inference_softmax_layer_call_fn_17584

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_14831`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
П
$__inference_BN3__layer_call_fn_17089

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN3__layer_call_and_return_conditional_losses_14625x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ј

љ
@__inference_CNN4__layer_call_and_return_conditional_losses_14653

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ж
?__inference_BN4__layer_call_and_return_conditional_losses_17244

inputs)
sub_readvariableop_resource:0%
readvariableop_resource:0'
readvariableop_1_resource:0+
add_1_readvariableop_resource:0
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:0*
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
:0:
SqrtSqrtadd:z:0*
T0*
_output_shapes
:0`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ0n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:0*
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ0a
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ0
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ0: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ъ
f
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_14808

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ`   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ`:W S
/
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Ї
П
$__inference_BN5__layer_call_fn_17335

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN5__layer_call_and_return_conditional_losses_14731w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@@
 
_user_specified_nameinputs
Ѓ

љ
@__inference_CNN5__layer_call_and_return_conditional_losses_17322

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@0
 
_user_specified_nameinputs
Й
K
/__inference_max_pooling2d_3_layer_call_fn_17298

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_14450
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
П
$__inference_BN1__layer_call_fn_16843

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN1__layer_call_and_return_conditional_losses_14519x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё*
Ъ
?__inference_BN2__layer_call_and_return_conditional_losses_17037

inputs%
readvariableop_resource:'
readvariableop_2_resource:'
readvariableop_4_resource:+
add_3_readvariableop_resource:
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
:E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
:*
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
:L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
:I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
:*
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџL
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
:<
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:*
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
N
"__inference__update_step_xla_16806
gradient
variable:`*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:`: *
	_noinline(:H D

_output_shapes

:`
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Я
V
"__inference__update_step_xla_16766
gradient"
variable:0@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:0@: *
	_noinline(:P L
&
_output_shapes
:0@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Е
I
-__inference_max_pooling2d_layer_call_fn_16929

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_14414
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16761
gradient
variable:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:0: *
	_noinline(:D @

_output_shapes
:0
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ё*
Ъ
?__inference_BN3__layer_call_and_return_conditional_losses_15256

inputs%
readvariableop_resource: '
readvariableop_2_resource: '
readvariableop_4_resource: +
add_3_readvariableop_resource: 
identityЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2ЂReadVariableOp_3ЂReadVariableOp_4Ђadd_3/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          І
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
: *
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
: L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=]
mul_1Mulmoments/Squeeze:output:0mul_1/y:output:0*
T0*
_output_shapes
: E
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: И
AssignVariableOpAssignVariableOpreadvariableop_resourceadd:z:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(w
ReadVariableOp_1ReadVariableOpreadvariableop_resource^AssignVariableOp*
_output_shapes
: *
dtype0И
AssignVariableOp_1AssignVariableOpreadvariableop_resourceReadVariableOp_1:value:0^AssignVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0L
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?]
mul_2MulReadVariableOp_2:value:0mul_2/y:output:0*
T0*
_output_shapes
: L
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=_
mul_3Mulmoments/Squeeze_1:output:0mul_3/y:output:0*
T0*
_output_shapes
: I
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: Р
AssignVariableOp_2AssignVariableOpreadvariableop_2_resource	add_1:z:0^ReadVariableOp_2*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape({
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource^AssignVariableOp_2*
_output_shapes
: *
dtype0М
AssignVariableOp_3AssignVariableOpreadvariableop_2_resourceReadVariableOp_3:value:0^AssignVariableOp_2^ReadVariableOp_3*
_output_shapes
 *
dtype0*
validate_shape(g
subSubinputsmoments/Squeeze:output:0*
T0*0
_output_shapes
:џџџџџџџџџ L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75a
add_2AddV2moments/Squeeze_1:output:0add_2/y:output:0*
T0*
_output_shapes
: <
SqrtSqrt	add_2:z:0*
T0*
_output_shapes
: `
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџ f
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
: *
dtype0n
mul_4MulReadVariableOp_4:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџ n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
: *
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42,
add_3/ReadVariableOpadd_3/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

C
'__inference_flatten_layer_call_fn_17594

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_14839`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16711
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
K
/__inference_max_pooling2d_4_layer_call_fn_17421

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_14462
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_16751
gradient
variable:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:0: *
	_noinline(:D @

_output_shapes
:0
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ј

љ
@__inference_CNN1__layer_call_and_return_conditional_losses_14494

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т	
№
?__inference_FC1__layer_call_and_return_conditional_losses_17579

inputs0
matmul_readvariableop_resource:`-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_16686
gradient"
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:: *
	_noinline(:P L
&
_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ЙЈ
,
!__inference__traced_restore_18053
file_prefix7
assignvariableop_cnn1__kernel:+
assignvariableop_1_cnn1__bias:7
)assignvariableop_2_bn1__custom_batch_beta:8
*assignvariableop_3_bn1__custom_batch_gamma:>
0assignvariableop_4_bn1__custom_batch_moving_mean:B
4assignvariableop_5_bn1__custom_batch_moving_variance:9
assignvariableop_6_cnn2__kernel:+
assignvariableop_7_cnn2__bias:7
)assignvariableop_8_bn2__custom_batch_beta:8
*assignvariableop_9_bn2__custom_batch_gamma:?
1assignvariableop_10_bn2__custom_batch_moving_mean:C
5assignvariableop_11_bn2__custom_batch_moving_variance::
 assignvariableop_12_cnn3__kernel: ,
assignvariableop_13_cnn3__bias: 8
*assignvariableop_14_bn3__custom_batch_beta: 9
+assignvariableop_15_bn3__custom_batch_gamma: ?
1assignvariableop_16_bn3__custom_batch_moving_mean: C
5assignvariableop_17_bn3__custom_batch_moving_variance: :
 assignvariableop_18_cnn4__kernel: 0,
assignvariableop_19_cnn4__bias:08
*assignvariableop_20_bn4__custom_batch_beta:09
+assignvariableop_21_bn4__custom_batch_gamma:0?
1assignvariableop_22_bn4__custom_batch_moving_mean:0C
5assignvariableop_23_bn4__custom_batch_moving_variance:0:
 assignvariableop_24_cnn5__kernel:0@,
assignvariableop_25_cnn5__bias:@8
*assignvariableop_26_bn5__custom_batch_beta:@9
+assignvariableop_27_bn5__custom_batch_gamma:@?
1assignvariableop_28_bn5__custom_batch_moving_mean:@C
5assignvariableop_29_bn5__custom_batch_moving_variance:@:
 assignvariableop_30_cnn6__kernel:@`,
assignvariableop_31_cnn6__bias:`8
*assignvariableop_32_bn6__custom_batch_beta:`9
+assignvariableop_33_bn6__custom_batch_gamma:`?
1assignvariableop_34_bn6__custom_batch_moving_mean:`C
5assignvariableop_35_bn6__custom_batch_moving_variance:`1
assignvariableop_36_fc1__kernel:`+
assignvariableop_37_fc1__bias:'
assignvariableop_38_iteration:	 +
!assignvariableop_39_learning_rate: @
&assignvariableop_40_sgd_m_cnn1__kernel:2
$assignvariableop_41_sgd_m_cnn1__bias:>
0assignvariableop_42_sgd_m_bn1__custom_batch_beta:?
1assignvariableop_43_sgd_m_bn1__custom_batch_gamma:@
&assignvariableop_44_sgd_m_cnn2__kernel:2
$assignvariableop_45_sgd_m_cnn2__bias:>
0assignvariableop_46_sgd_m_bn2__custom_batch_beta:?
1assignvariableop_47_sgd_m_bn2__custom_batch_gamma:@
&assignvariableop_48_sgd_m_cnn3__kernel: 2
$assignvariableop_49_sgd_m_cnn3__bias: >
0assignvariableop_50_sgd_m_bn3__custom_batch_beta: ?
1assignvariableop_51_sgd_m_bn3__custom_batch_gamma: @
&assignvariableop_52_sgd_m_cnn4__kernel: 02
$assignvariableop_53_sgd_m_cnn4__bias:0>
0assignvariableop_54_sgd_m_bn4__custom_batch_beta:0?
1assignvariableop_55_sgd_m_bn4__custom_batch_gamma:0@
&assignvariableop_56_sgd_m_cnn5__kernel:0@2
$assignvariableop_57_sgd_m_cnn5__bias:@>
0assignvariableop_58_sgd_m_bn5__custom_batch_beta:@?
1assignvariableop_59_sgd_m_bn5__custom_batch_gamma:@@
&assignvariableop_60_sgd_m_cnn6__kernel:@`2
$assignvariableop_61_sgd_m_cnn6__bias:`>
0assignvariableop_62_sgd_m_bn6__custom_batch_beta:`?
1assignvariableop_63_sgd_m_bn6__custom_batch_gamma:`7
%assignvariableop_64_sgd_m_fc1__kernel:`1
#assignvariableop_65_sgd_m_fc1__bias:%
assignvariableop_66_total_1: %
assignvariableop_67_count_1: #
assignvariableop_68_total: #
assignvariableop_69_count: 
identity_71ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Я!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*ѕ 
valueы Bш GB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-1/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-1/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-3/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-3/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-3/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-3/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-5/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-5/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-5/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-5/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-7/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-7/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-7/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-7/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-9/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-9/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBHlayer_with_weights-9/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBLlayer_with_weights-9/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-11/custom_batch_beta/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-11/custom_batch_gamma/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-11/custom_batch_moving_mean/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-11/custom_batch_moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:G*
dtype0*Ѓ
valueBGB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*В
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*U
dtypesK
I2G	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOpAssignVariableOpassignvariableop_cnn1__kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_1AssignVariableOpassignvariableop_1_cnn1__biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOp)assignvariableop_2_bn1__custom_batch_betaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_3AssignVariableOp*assignvariableop_3_bn1__custom_batch_gammaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_4AssignVariableOp0assignvariableop_4_bn1__custom_batch_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_5AssignVariableOp4assignvariableop_5_bn1__custom_batch_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_6AssignVariableOpassignvariableop_6_cnn2__kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_7AssignVariableOpassignvariableop_7_cnn2__biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_8AssignVariableOp)assignvariableop_8_bn2__custom_batch_betaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_9AssignVariableOp*assignvariableop_9_bn2__custom_batch_gammaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_10AssignVariableOp1assignvariableop_10_bn2__custom_batch_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_11AssignVariableOp5assignvariableop_11_bn2__custom_batch_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_12AssignVariableOp assignvariableop_12_cnn3__kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_13AssignVariableOpassignvariableop_13_cnn3__biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_14AssignVariableOp*assignvariableop_14_bn3__custom_batch_betaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_15AssignVariableOp+assignvariableop_15_bn3__custom_batch_gammaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_16AssignVariableOp1assignvariableop_16_bn3__custom_batch_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_17AssignVariableOp5assignvariableop_17_bn3__custom_batch_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_18AssignVariableOp assignvariableop_18_cnn4__kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_19AssignVariableOpassignvariableop_19_cnn4__biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_20AssignVariableOp*assignvariableop_20_bn4__custom_batch_betaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_21AssignVariableOp+assignvariableop_21_bn4__custom_batch_gammaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_22AssignVariableOp1assignvariableop_22_bn4__custom_batch_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_23AssignVariableOp5assignvariableop_23_bn4__custom_batch_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_24AssignVariableOp assignvariableop_24_cnn5__kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_25AssignVariableOpassignvariableop_25_cnn5__biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_26AssignVariableOp*assignvariableop_26_bn5__custom_batch_betaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_27AssignVariableOp+assignvariableop_27_bn5__custom_batch_gammaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_28AssignVariableOp1assignvariableop_28_bn5__custom_batch_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_29AssignVariableOp5assignvariableop_29_bn5__custom_batch_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_30AssignVariableOp assignvariableop_30_cnn6__kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_31AssignVariableOpassignvariableop_31_cnn6__biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_32AssignVariableOp*assignvariableop_32_bn6__custom_batch_betaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_33AssignVariableOp+assignvariableop_33_bn6__custom_batch_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_34AssignVariableOp1assignvariableop_34_bn6__custom_batch_moving_meanIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_35AssignVariableOp5assignvariableop_35_bn6__custom_batch_moving_varianceIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_36AssignVariableOpassignvariableop_36_fc1__kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_37AssignVariableOpassignvariableop_37_fc1__biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_38AssignVariableOpassignvariableop_38_iterationIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_39AssignVariableOp!assignvariableop_39_learning_rateIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_40AssignVariableOp&assignvariableop_40_sgd_m_cnn1__kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_41AssignVariableOp$assignvariableop_41_sgd_m_cnn1__biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_42AssignVariableOp0assignvariableop_42_sgd_m_bn1__custom_batch_betaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_43AssignVariableOp1assignvariableop_43_sgd_m_bn1__custom_batch_gammaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_44AssignVariableOp&assignvariableop_44_sgd_m_cnn2__kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_45AssignVariableOp$assignvariableop_45_sgd_m_cnn2__biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_46AssignVariableOp0assignvariableop_46_sgd_m_bn2__custom_batch_betaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_47AssignVariableOp1assignvariableop_47_sgd_m_bn2__custom_batch_gammaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_48AssignVariableOp&assignvariableop_48_sgd_m_cnn3__kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_49AssignVariableOp$assignvariableop_49_sgd_m_cnn3__biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_50AssignVariableOp0assignvariableop_50_sgd_m_bn3__custom_batch_betaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_51AssignVariableOp1assignvariableop_51_sgd_m_bn3__custom_batch_gammaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_52AssignVariableOp&assignvariableop_52_sgd_m_cnn4__kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_53AssignVariableOp$assignvariableop_53_sgd_m_cnn4__biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_m_bn4__custom_batch_betaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_55AssignVariableOp1assignvariableop_55_sgd_m_bn4__custom_batch_gammaIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_56AssignVariableOp&assignvariableop_56_sgd_m_cnn5__kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_57AssignVariableOp$assignvariableop_57_sgd_m_cnn5__biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_58AssignVariableOp0assignvariableop_58_sgd_m_bn5__custom_batch_betaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_59AssignVariableOp1assignvariableop_59_sgd_m_bn5__custom_batch_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_60AssignVariableOp&assignvariableop_60_sgd_m_cnn6__kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_61AssignVariableOp$assignvariableop_61_sgd_m_cnn6__biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_62AssignVariableOp0assignvariableop_62_sgd_m_bn6__custom_batch_betaIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_63AssignVariableOp1assignvariableop_63_sgd_m_bn6__custom_batch_gammaIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_64AssignVariableOp%assignvariableop_64_sgd_m_fc1__kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_65AssignVariableOp#assignvariableop_65_sgd_m_fc1__biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_66AssignVariableOpassignvariableop_66_total_1Identity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_67AssignVariableOpassignvariableop_67_count_1Identity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_68AssignVariableOpassignvariableop_68_totalIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_69AssignVariableOpassignvariableop_69_countIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 г
Identity_70Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_71IdentityIdentity_70:output:0^NoOp_1*
T0*
_output_shapes
: Р
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_71Identity_71:output:0*Ѓ
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

ж
?__inference_BN1__layer_call_and_return_conditional_losses_16875

inputs)
sub_readvariableop_resource:%
readvariableop_resource:'
readvariableop_1_resource:+
add_1_readvariableop_resource:
identityЂReadVariableOpЂReadVariableOp_1Ђadd_1/ReadVariableOpЂsub/ReadVariableOpj
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype0i
subSubinputssub/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Н75Y
addAddV2ReadVariableOp:value:0add/y:output:0*
T0*
_output_shapes
::
SqrtSqrtadd:z:0*
T0*
_output_shapes
:`
truedivRealDivsub:z:0Sqrt:y:0*
T0*0
_output_shapes
:џџџџџџџџџf
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0l
mulMulReadVariableOp_1:value:0truediv:z:0*
T0*0
_output_shapes
:џџџџџџџџџn
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype0p
add_1AddV2mul:z:0add_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџa
IdentityIdentity	add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџ: : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_1/ReadVariableOpadd_1/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Г
serving_default
D
input_19
serving_default_input_1:0џџџџџџџџџ;
flatten0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:в
Ф
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer_with_weights-8
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer-25
layer_with_weights-12
layer-26
layer-27
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%	optimizer
&
signatures"
_tf_keras_network
"
_tf_keras_input_layer
н
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op"
_tf_keras_layer
Я
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6custom_batch_beta
6beta
7custom_batch_gamma
	7gamma
8custom_batch_moving_mean
8moving_mean
 9custom_batch_moving_variance
9moving_variance"
_tf_keras_layer
Ѕ
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
н
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op"
_tf_keras_layer
Я
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Ucustom_batch_beta
Ubeta
Vcustom_batch_gamma
	Vgamma
Wcustom_batch_moving_mean
Wmoving_mean
 Xcustom_batch_moving_variance
Xmoving_variance"
_tf_keras_layer
Ѕ
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
н
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op"
_tf_keras_layer
Я
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
tcustom_batch_beta
tbeta
ucustom_batch_gamma
	ugamma
vcustom_batch_moving_mean
vmoving_mean
 wcustom_batch_moving_variance
wmoving_variance"
_tf_keras_layer
Ѕ
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
Љ
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
custom_batch_beta
	beta
custom_batch_gamma

gamma
custom_batch_moving_mean
moving_mean
!custom_batch_moving_variance
moving_variance"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses
Љkernel
	Њbias
!Ћ_jit_compiled_convolution_op"
_tf_keras_layer
н
Ќ	variables
­trainable_variables
Ўregularization_losses
Џ	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вcustom_batch_beta
	Вbeta
Гcustom_batch_gamma

Гgamma
Дcustom_batch_moving_mean
Дmoving_mean
!Еcustom_batch_moving_variance
Еmoving_variance"
_tf_keras_layer
Ћ
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Шkernel
	Щbias
!Ъ_jit_compiled_convolution_op"
_tf_keras_layer
н
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses
бcustom_batch_beta
	бbeta
вcustom_batch_gamma

вgamma
гcustom_batch_moving_mean
гmoving_mean
!дcustom_batch_moving_variance
дmoving_variance"
_tf_keras_layer
Ћ
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
л	variables
мtrainable_variables
нregularization_losses
о	keras_api
п__call__
+р&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"
_tf_keras_layer
У
ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses
эkernel
	юbias"
_tf_keras_layer
Ћ
я	variables
№trainable_variables
ёregularization_losses
ђ	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses"
_tf_keras_layer
к
-0
.1
62
73
84
95
L6
M7
U8
V9
W10
X11
k12
l13
t14
u15
v16
w17
18
19
20
21
22
23
Љ24
Њ25
В26
Г27
Д28
Е29
Ш30
Щ31
б32
в33
г34
д35
э36
ю37"
trackable_list_wrapper
є
-0
.1
62
73
L4
M5
U6
V7
k8
l9
t10
u11
12
13
14
15
Љ16
Њ17
В18
Г19
Ш20
Щ21
б22
в23
э24
ю25"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
б
trace_0
trace_1
trace_2
trace_32о
%__inference_model_layer_call_fn_14921
%__inference_model_layer_call_fn_16174
%__inference_model_layer_call_fn_16255
%__inference_model_layer_call_fn_15792П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
Н
trace_0
trace_1
trace_2
trace_32Ъ
@__inference_model_layer_call_and_return_conditional_losses_16408
@__inference_model_layer_call_and_return_conditional_losses_16681
@__inference_model_layer_call_and_return_conditional_losses_15900
@__inference_model_layer_call_and_return_conditional_losses_16008П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ЫBШ
 __inference__wrapped_model_14405input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 


_variables
_iterations
_learning_rate
_index_dict
	momentums
_update_step_xla"
experimentalOptimizer
-
serving_default"
signature_map
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
%__inference_CNN1__layer_call_fn_16820Ђ
В
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
annotationsЊ *
 ztrace_0

trace_02ч
@__inference_CNN1__layer_call_and_return_conditional_losses_16830Ђ
В
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
annotationsЊ *
 ztrace_0
&:$2CNN1_/kernel
:2
CNN1_/bias
Д2БЎ
ЃВ
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
annotationsЊ *
 0
<
60
71
82
93"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
И
trace_0
trace_12§
$__inference_BN1__layer_call_fn_16843
$__inference_BN1__layer_call_fn_16856Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ю
trace_0
trace_12Г
?__inference_BN1__layer_call_and_return_conditional_losses_16875
?__inference_BN1__layer_call_and_return_conditional_losses_16914Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
$:"2BN1_/custom_batch_beta
%:#2BN1_/custom_batch_gamma
):'2BN1_/custom_batch_moving_mean
-:+2!BN1_/custom_batch_moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ы
Єtrace_02Ь
%__inference_re_lu_layer_call_fn_16919Ђ
В
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
annotationsЊ *
 zЄtrace_0

Ѕtrace_02ч
@__inference_re_lu_layer_call_and_return_conditional_losses_16924Ђ
В
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
annotationsЊ *
 zЅtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ѓ
Ћtrace_02д
-__inference_max_pooling2d_layer_call_fn_16929Ђ
В
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
annotationsЊ *
 zЋtrace_0

Ќtrace_02я
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_16934Ђ
В
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
annotationsЊ *
 zЌtrace_0
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
­non_trainable_variables
Ўlayers
Џmetrics
 Аlayer_regularization_losses
Бlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
ы
Вtrace_02Ь
%__inference_CNN2__layer_call_fn_16943Ђ
В
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
annotationsЊ *
 zВtrace_0

Гtrace_02ч
@__inference_CNN2__layer_call_and_return_conditional_losses_16953Ђ
В
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
annotationsЊ *
 zГtrace_0
&:$2CNN2_/kernel
:2
CNN2_/bias
Д2БЎ
ЃВ
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
annotationsЊ *
 0
<
U0
V1
W2
X3"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
И
Йtrace_0
Кtrace_12§
$__inference_BN2__layer_call_fn_16966
$__inference_BN2__layer_call_fn_16979Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЙtrace_0zКtrace_1
ю
Лtrace_0
Мtrace_12Г
?__inference_BN2__layer_call_and_return_conditional_losses_16998
?__inference_BN2__layer_call_and_return_conditional_losses_17037Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0zМtrace_1
$:"2BN2_/custom_batch_beta
%:#2BN2_/custom_batch_gamma
):'2BN2_/custom_batch_moving_mean
-:+2!BN2_/custom_batch_moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
э
Тtrace_02Ю
'__inference_re_lu_1_layer_call_fn_17042Ђ
В
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
annotationsЊ *
 zТtrace_0

Уtrace_02щ
B__inference_re_lu_1_layer_call_and_return_conditional_losses_17047Ђ
В
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
annotationsЊ *
 zУtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
ѕ
Щtrace_02ж
/__inference_max_pooling2d_1_layer_call_fn_17052Ђ
В
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
annotationsЊ *
 zЩtrace_0

Ъtrace_02ё
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17057Ђ
В
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
annotationsЊ *
 zЪtrace_0
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ы
аtrace_02Ь
%__inference_CNN3__layer_call_fn_17066Ђ
В
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
annotationsЊ *
 zаtrace_0

бtrace_02ч
@__inference_CNN3__layer_call_and_return_conditional_losses_17076Ђ
В
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
annotationsЊ *
 zбtrace_0
&:$ 2CNN3_/kernel
: 2
CNN3_/bias
Д2БЎ
ЃВ
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
annotationsЊ *
 0
<
t0
u1
v2
w3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
И
зtrace_0
иtrace_12§
$__inference_BN3__layer_call_fn_17089
$__inference_BN3__layer_call_fn_17102Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zзtrace_0zиtrace_1
ю
йtrace_0
кtrace_12Г
?__inference_BN3__layer_call_and_return_conditional_losses_17121
?__inference_BN3__layer_call_and_return_conditional_losses_17160Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zйtrace_0zкtrace_1
$:" 2BN3_/custom_batch_beta
%:# 2BN3_/custom_batch_gamma
):' 2BN3_/custom_batch_moving_mean
-:+ 2!BN3_/custom_batch_moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
э
рtrace_02Ю
'__inference_re_lu_2_layer_call_fn_17165Ђ
В
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
annotationsЊ *
 zрtrace_0

сtrace_02щ
B__inference_re_lu_2_layer_call_and_return_conditional_losses_17170Ђ
В
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
annotationsЊ *
 zсtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ж
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ѕ
чtrace_02ж
/__inference_max_pooling2d_2_layer_call_fn_17175Ђ
В
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
annotationsЊ *
 zчtrace_0

шtrace_02ё
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17180Ђ
В
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
annotationsЊ *
 zшtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ы
юtrace_02Ь
%__inference_CNN4__layer_call_fn_17189Ђ
В
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
annotationsЊ *
 zюtrace_0

яtrace_02ч
@__inference_CNN4__layer_call_and_return_conditional_losses_17199Ђ
В
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
annotationsЊ *
 zяtrace_0
&:$ 02CNN4_/kernel
:02
CNN4_/bias
Д2БЎ
ЃВ
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
annotationsЊ *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
И
ѕtrace_0
іtrace_12§
$__inference_BN4__layer_call_fn_17212
$__inference_BN4__layer_call_fn_17225Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѕtrace_0zіtrace_1
ю
їtrace_0
јtrace_12Г
?__inference_BN4__layer_call_and_return_conditional_losses_17244
?__inference_BN4__layer_call_and_return_conditional_losses_17283Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zїtrace_0zјtrace_1
$:"02BN4_/custom_batch_beta
%:#02BN4_/custom_batch_gamma
):'02BN4_/custom_batch_moving_mean
-:+02!BN4_/custom_batch_moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
э
ўtrace_02Ю
'__inference_re_lu_3_layer_call_fn_17288Ђ
В
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
annotationsЊ *
 zўtrace_0

џtrace_02щ
B__inference_re_lu_3_layer_call_and_return_conditional_losses_17293Ђ
В
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
annotationsЊ *
 zџtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
ѕ
trace_02ж
/__inference_max_pooling2d_3_layer_call_fn_17298Ђ
В
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
annotationsЊ *
 ztrace_0

trace_02ё
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17303Ђ
В
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
annotationsЊ *
 ztrace_0
0
Љ0
Њ1"
trackable_list_wrapper
0
Љ0
Њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
%__inference_CNN5__layer_call_fn_17312Ђ
В
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
annotationsЊ *
 ztrace_0

trace_02ч
@__inference_CNN5__layer_call_and_return_conditional_losses_17322Ђ
В
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
annotationsЊ *
 ztrace_0
&:$0@2CNN5_/kernel
:@2
CNN5_/bias
Д2БЎ
ЃВ
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
annotationsЊ *
 0
@
В0
Г1
Д2
Е3"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ќ	variables
­trainable_variables
Ўregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
И
trace_0
trace_12§
$__inference_BN5__layer_call_fn_17335
$__inference_BN5__layer_call_fn_17348Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ю
trace_0
trace_12Г
?__inference_BN5__layer_call_and_return_conditional_losses_17367
?__inference_BN5__layer_call_and_return_conditional_losses_17406Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
$:"@2BN5_/custom_batch_beta
%:#@2BN5_/custom_batch_gamma
):'@2BN5_/custom_batch_moving_mean
-:+@2!BN5_/custom_batch_moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_re_lu_4_layer_call_fn_17411Ђ
В
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
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_re_lu_4_layer_call_and_return_conditional_losses_17416Ђ
В
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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
ѕ
Ѓtrace_02ж
/__inference_max_pooling2d_4_layer_call_fn_17421Ђ
В
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
annotationsЊ *
 zЃtrace_0

Єtrace_02ё
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17426Ђ
В
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
annotationsЊ *
 zЄtrace_0
0
Ш0
Щ1"
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
ы
Њtrace_02Ь
%__inference_CNN6__layer_call_fn_17435Ђ
В
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
annotationsЊ *
 zЊtrace_0

Ћtrace_02ч
@__inference_CNN6__layer_call_and_return_conditional_losses_17445Ђ
В
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
annotationsЊ *
 zЋtrace_0
&:$@`2CNN6_/kernel
:`2
CNN6_/bias
Д2БЎ
ЃВ
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
annotationsЊ *
 0
@
б0
в1
г2
д3"
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
И
Бtrace_0
Вtrace_12§
$__inference_BN6__layer_call_fn_17458
$__inference_BN6__layer_call_fn_17471Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0zВtrace_1
ю
Гtrace_0
Дtrace_12Г
?__inference_BN6__layer_call_and_return_conditional_losses_17490
?__inference_BN6__layer_call_and_return_conditional_losses_17529Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zГtrace_0zДtrace_1
$:"`2BN6_/custom_batch_beta
%:#`2BN6_/custom_batch_gamma
):'`2BN6_/custom_batch_moving_mean
-:+`2!BN6_/custom_batch_moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
э
Кtrace_02Ю
'__inference_re_lu_5_layer_call_fn_17534Ђ
В
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
annotationsЊ *
 zКtrace_0

Лtrace_02щ
B__inference_re_lu_5_layer_call_and_return_conditional_losses_17539Ђ
В
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
annotationsЊ *
 zЛtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
л	variables
мtrainable_variables
нregularization_losses
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
ї
Сtrace_02и
1__inference_average_pooling2d_layer_call_fn_17544Ђ
В
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
annotationsЊ *
 zСtrace_0

Тtrace_02ѓ
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_17549Ђ
В
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
annotationsЊ *
 zТtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
ѕ
Шtrace_02ж
/__inference_FC1_preFlatten1_layer_call_fn_17554Ђ
В
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
annotationsЊ *
 zШtrace_0

Щtrace_02ё
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_17560Ђ
В
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
annotationsЊ *
 zЩtrace_0
0
э0
ю1"
trackable_list_wrapper
0
э0
ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
ъ
Яtrace_02Ы
$__inference_FC1__layer_call_fn_17569Ђ
В
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
annotationsЊ *
 zЯtrace_0

аtrace_02ц
?__inference_FC1__layer_call_and_return_conditional_losses_17579Ђ
В
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
annotationsЊ *
 zаtrace_0
:`2FC1_/kernel
:2	FC1_/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
я	variables
№trainable_variables
ёregularization_losses
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
њ
жtrace_02л
'__inference_softmax_layer_call_fn_17584Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zжtrace_0

зtrace_02і
B__inference_softmax_layer_call_and_return_conditional_losses_17589Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zзtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
э
нtrace_02Ю
'__inference_flatten_layer_call_fn_17594Ђ
В
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
annotationsЊ *
 zнtrace_0

оtrace_02щ
B__inference_flatten_layer_call_and_return_conditional_losses_17600Ђ
В
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
annotationsЊ *
 zоtrace_0
|
80
91
W2
X3
v4
w5
6
7
Д8
Е9
г10
д11"
trackable_list_wrapper
ў
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
0
п0
р1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
їBє
%__inference_model_layer_call_fn_14921input_1"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
%__inference_model_layer_call_fn_16174inputs"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
%__inference_model_layer_call_fn_16255inputs"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
%__inference_model_layer_call_fn_15792input_1"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_16408inputs"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_16681inputs"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_15900input_1"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
@__inference_model_layer_call_and_return_conditional_losses_16008input_1"П
ЖВВ
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

0
с1
т2
у3
ф4
х5
ц6
ч7
ш8
щ9
ъ10
ы11
ь12
э13
ю14
я15
№16
ё17
ђ18
ѓ19
є20
ѕ21
і22
ї23
ј24
љ25
њ26"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper

с0
т1
у2
ф3
х4
ц5
ч6
ш7
щ8
ъ9
ы10
ь11
э12
ю13
я14
№15
ё16
ђ17
ѓ18
є19
ѕ20
і21
ї22
ј23
љ24
њ25"
trackable_list_wrapper
п
ћtrace_0
ќtrace_1
§trace_2
ўtrace_3
џtrace_4
trace_5
trace_6
trace_7
trace_8
trace_9
trace_10
trace_11
trace_12
trace_13
trace_14
trace_15
trace_16
trace_17
trace_18
trace_19
trace_20
trace_21
trace_22
trace_23
trace_24
trace_252ф
"__inference__update_step_xla_16686
"__inference__update_step_xla_16691
"__inference__update_step_xla_16696
"__inference__update_step_xla_16701
"__inference__update_step_xla_16706
"__inference__update_step_xla_16711
"__inference__update_step_xla_16716
"__inference__update_step_xla_16721
"__inference__update_step_xla_16726
"__inference__update_step_xla_16731
"__inference__update_step_xla_16736
"__inference__update_step_xla_16741
"__inference__update_step_xla_16746
"__inference__update_step_xla_16751
"__inference__update_step_xla_16756
"__inference__update_step_xla_16761
"__inference__update_step_xla_16766
"__inference__update_step_xla_16771
"__inference__update_step_xla_16776
"__inference__update_step_xla_16781
"__inference__update_step_xla_16786
"__inference__update_step_xla_16791
"__inference__update_step_xla_16796
"__inference__update_step_xla_16801
"__inference__update_step_xla_16806
"__inference__update_step_xla_16811Й
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zћtrace_0zќtrace_1z§trace_2zўtrace_3zџtrace_4ztrace_5ztrace_6ztrace_7ztrace_8ztrace_9ztrace_10ztrace_11ztrace_12ztrace_13ztrace_14ztrace_15ztrace_16ztrace_17ztrace_18ztrace_19ztrace_20ztrace_21ztrace_22ztrace_23ztrace_24ztrace_25
ЪBЧ
#__inference_signature_wrapper_16093input_1"
В
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
annotationsЊ *
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
йBж
%__inference_CNN1__layer_call_fn_16820inputs"Ђ
В
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
annotationsЊ *
 
єBё
@__inference_CNN1__layer_call_and_return_conditional_losses_16830inputs"Ђ
В
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
annotationsЊ *
 
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
$__inference_BN1__layer_call_fn_16843inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
$__inference_BN1__layer_call_fn_16856inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN1__layer_call_and_return_conditional_losses_16875inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN1__layer_call_and_return_conditional_losses_16914inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
йBж
%__inference_re_lu_layer_call_fn_16919inputs"Ђ
В
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
annotationsЊ *
 
єBё
@__inference_re_lu_layer_call_and_return_conditional_losses_16924inputs"Ђ
В
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
annotationsЊ *
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
сBо
-__inference_max_pooling2d_layer_call_fn_16929inputs"Ђ
В
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
annotationsЊ *
 
ќBљ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_16934inputs"Ђ
В
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
annotationsЊ *
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
йBж
%__inference_CNN2__layer_call_fn_16943inputs"Ђ
В
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
annotationsЊ *
 
єBё
@__inference_CNN2__layer_call_and_return_conditional_losses_16953inputs"Ђ
В
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
annotationsЊ *
 
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
$__inference_BN2__layer_call_fn_16966inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
$__inference_BN2__layer_call_fn_16979inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN2__layer_call_and_return_conditional_losses_16998inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN2__layer_call_and_return_conditional_losses_17037inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
лBи
'__inference_re_lu_1_layer_call_fn_17042inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_re_lu_1_layer_call_and_return_conditional_losses_17047inputs"Ђ
В
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
annotationsЊ *
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
уBр
/__inference_max_pooling2d_1_layer_call_fn_17052inputs"Ђ
В
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
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17057inputs"Ђ
В
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
annotationsЊ *
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
йBж
%__inference_CNN3__layer_call_fn_17066inputs"Ђ
В
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
annotationsЊ *
 
єBё
@__inference_CNN3__layer_call_and_return_conditional_losses_17076inputs"Ђ
В
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
annotationsЊ *
 
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
$__inference_BN3__layer_call_fn_17089inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
$__inference_BN3__layer_call_fn_17102inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN3__layer_call_and_return_conditional_losses_17121inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN3__layer_call_and_return_conditional_losses_17160inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
лBи
'__inference_re_lu_2_layer_call_fn_17165inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_re_lu_2_layer_call_and_return_conditional_losses_17170inputs"Ђ
В
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
annotationsЊ *
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
уBр
/__inference_max_pooling2d_2_layer_call_fn_17175inputs"Ђ
В
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
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17180inputs"Ђ
В
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
annotationsЊ *
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
йBж
%__inference_CNN4__layer_call_fn_17189inputs"Ђ
В
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
annotationsЊ *
 
єBё
@__inference_CNN4__layer_call_and_return_conditional_losses_17199inputs"Ђ
В
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
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
$__inference_BN4__layer_call_fn_17212inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
$__inference_BN4__layer_call_fn_17225inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN4__layer_call_and_return_conditional_losses_17244inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN4__layer_call_and_return_conditional_losses_17283inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
лBи
'__inference_re_lu_3_layer_call_fn_17288inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_re_lu_3_layer_call_and_return_conditional_losses_17293inputs"Ђ
В
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
annotationsЊ *
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
уBр
/__inference_max_pooling2d_3_layer_call_fn_17298inputs"Ђ
В
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
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17303inputs"Ђ
В
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
annotationsЊ *
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
йBж
%__inference_CNN5__layer_call_fn_17312inputs"Ђ
В
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
annotationsЊ *
 
єBё
@__inference_CNN5__layer_call_and_return_conditional_losses_17322inputs"Ђ
В
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
annotationsЊ *
 
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
$__inference_BN5__layer_call_fn_17335inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
$__inference_BN5__layer_call_fn_17348inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN5__layer_call_and_return_conditional_losses_17367inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN5__layer_call_and_return_conditional_losses_17406inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
лBи
'__inference_re_lu_4_layer_call_fn_17411inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_re_lu_4_layer_call_and_return_conditional_losses_17416inputs"Ђ
В
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
annotationsЊ *
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
уBр
/__inference_max_pooling2d_4_layer_call_fn_17421inputs"Ђ
В
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
annotationsЊ *
 
ўBћ
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17426inputs"Ђ
В
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
annotationsЊ *
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
йBж
%__inference_CNN6__layer_call_fn_17435inputs"Ђ
В
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
annotationsЊ *
 
єBё
@__inference_CNN6__layer_call_and_return_conditional_losses_17445inputs"Ђ
В
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
annotationsЊ *
 
0
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
$__inference_BN6__layer_call_fn_17458inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
$__inference_BN6__layer_call_fn_17471inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN6__layer_call_and_return_conditional_losses_17490inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
?__inference_BN6__layer_call_and_return_conditional_losses_17529inputs"Ў
ЅВЁ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
лBи
'__inference_re_lu_5_layer_call_fn_17534inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_re_lu_5_layer_call_and_return_conditional_losses_17539inputs"Ђ
В
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
annotationsЊ *
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
хBт
1__inference_average_pooling2d_layer_call_fn_17544inputs"Ђ
В
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
annotationsЊ *
 
B§
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_17549inputs"Ђ
В
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
annotationsЊ *
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
уBр
/__inference_FC1_preFlatten1_layer_call_fn_17554inputs"Ђ
В
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
annotationsЊ *
 
ўBћ
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_17560inputs"Ђ
В
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
annotationsЊ *
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
иBе
$__inference_FC1__layer_call_fn_17569inputs"Ђ
В
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
annotationsЊ *
 
ѓB№
?__inference_FC1__layer_call_and_return_conditional_losses_17579inputs"Ђ
В
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
annotationsЊ *
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
шBх
'__inference_softmax_layer_call_fn_17584inputs"Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_softmax_layer_call_and_return_conditional_losses_17589inputs"Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
лBи
'__inference_flatten_layer_call_fn_17594inputs"Ђ
В
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
annotationsЊ *
 
іBѓ
B__inference_flatten_layer_call_and_return_conditional_losses_17600inputs"Ђ
В
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
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
*:(2SGD/m/CNN1_/kernel
:2SGD/m/CNN1_/bias
(:&2SGD/m/BN1_/custom_batch_beta
):'2SGD/m/BN1_/custom_batch_gamma
*:(2SGD/m/CNN2_/kernel
:2SGD/m/CNN2_/bias
(:&2SGD/m/BN2_/custom_batch_beta
):'2SGD/m/BN2_/custom_batch_gamma
*:( 2SGD/m/CNN3_/kernel
: 2SGD/m/CNN3_/bias
(:& 2SGD/m/BN3_/custom_batch_beta
):' 2SGD/m/BN3_/custom_batch_gamma
*:( 02SGD/m/CNN4_/kernel
:02SGD/m/CNN4_/bias
(:&02SGD/m/BN4_/custom_batch_beta
):'02SGD/m/BN4_/custom_batch_gamma
*:(0@2SGD/m/CNN5_/kernel
:@2SGD/m/CNN5_/bias
(:&@2SGD/m/BN5_/custom_batch_beta
):'@2SGD/m/BN5_/custom_batch_gamma
*:(@`2SGD/m/CNN6_/kernel
:`2SGD/m/CNN6_/bias
(:&`2SGD/m/BN6_/custom_batch_beta
):'`2SGD/m/BN6_/custom_batch_gamma
!:`2SGD/m/FC1_/kernel
:2SGD/m/FC1_/bias
їBє
"__inference__update_step_xla_16686gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16691gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16696gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16701gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16706gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16711gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16716gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16721gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16726gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16731gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16736gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16741gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16746gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16751gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16756gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16761gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16766gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16771gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16776gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16781gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16786gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16791gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16796gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16801gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16806gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_16811gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperО
?__inference_BN1__layer_call_and_return_conditional_losses_16875{8976<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 О
?__inference_BN1__layer_call_and_return_conditional_losses_16914{8976<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
$__inference_BN1__layer_call_fn_16843p8976<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "*'
unknownџџџџџџџџџ
$__inference_BN1__layer_call_fn_16856p8976<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "*'
unknownџџџџџџџџџО
?__inference_BN2__layer_call_and_return_conditional_losses_16998{WXVU<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 О
?__inference_BN2__layer_call_and_return_conditional_losses_17037{WXVU<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
$__inference_BN2__layer_call_fn_16966pWXVU<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p 
Њ "*'
unknownџџџџџџџџџ
$__inference_BN2__layer_call_fn_16979pWXVU<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ
p
Њ "*'
unknownџџџџџџџџџО
?__inference_BN3__layer_call_and_return_conditional_losses_17121{vwut<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ 
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ 
 О
?__inference_BN3__layer_call_and_return_conditional_losses_17160{vwut<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ 
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ 
 
$__inference_BN3__layer_call_fn_17089pvwut<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ 
p 
Њ "*'
unknownџџџџџџџџџ 
$__inference_BN3__layer_call_fn_17102pvwut<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ 
p
Њ "*'
unknownџџџџџџџџџ Т
?__inference_BN4__layer_call_and_return_conditional_losses_17244<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ0
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ0
 Т
?__inference_BN4__layer_call_and_return_conditional_losses_17283<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ0
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ0
 
$__inference_BN4__layer_call_fn_17212t<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ0
p 
Њ "*'
unknownџџџџџџџџџ0
$__inference_BN4__layer_call_fn_17225t<Ђ9
2Ђ/
)&
inputsџџџџџџџџџ0
p
Њ "*'
unknownџџџџџџџџџ0Р
?__inference_BN5__layer_call_and_return_conditional_losses_17367}ДЕГВ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@@
 Р
?__inference_BN5__layer_call_and_return_conditional_losses_17406}ДЕГВ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@@
 
$__inference_BN5__layer_call_fn_17335rДЕГВ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p 
Њ ")&
unknownџџџџџџџџџ@@
$__inference_BN5__layer_call_fn_17348rДЕГВ;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@@
p
Њ ")&
unknownџџџџџџџџџ@@Р
?__inference_BN6__layer_call_and_return_conditional_losses_17490}гдвб;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ `
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ `
 Р
?__inference_BN6__layer_call_and_return_conditional_losses_17529}гдвб;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ `
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ `
 
$__inference_BN6__layer_call_fn_17458rгдвб;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ `
p 
Њ ")&
unknownџџџџџџџџџ `
$__inference_BN6__layer_call_fn_17471rгдвб;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ `
p
Њ ")&
unknownџџџџџџџџџ `Й
@__inference_CNN1__layer_call_and_return_conditional_losses_16830u-.8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
%__inference_CNN1__layer_call_fn_16820j-.8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџЙ
@__inference_CNN2__layer_call_and_return_conditional_losses_16953uLM8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
%__inference_CNN2__layer_call_fn_16943jLM8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџЙ
@__inference_CNN3__layer_call_and_return_conditional_losses_17076ukl8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ 
 
%__inference_CNN3__layer_call_fn_17066jkl8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџ Л
@__inference_CNN4__layer_call_and_return_conditional_losses_17199w8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ0
 
%__inference_CNN4__layer_call_fn_17189l8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ 
Њ "*'
unknownџџџџџџџџџ0Й
@__inference_CNN5__layer_call_and_return_conditional_losses_17322uЉЊ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@0
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@@
 
%__inference_CNN5__layer_call_fn_17312jЉЊ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@0
Њ ")&
unknownџџџџџџџџџ@@Й
@__inference_CNN6__layer_call_and_return_conditional_losses_17445uШЩ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ @
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ `
 
%__inference_CNN6__layer_call_fn_17435jШЩ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ @
Њ ")&
unknownџџџџџџџџџ `Ј
?__inference_FC1__layer_call_and_return_conditional_losses_17579eэю/Ђ,
%Ђ"
 
inputsџџџџџџџџџ`
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
$__inference_FC1__layer_call_fn_17569Zэю/Ђ,
%Ђ"
 
inputsџџџџџџџџџ`
Њ "!
unknownџџџџџџџџџЕ
J__inference_FC1_preFlatten1_layer_call_and_return_conditional_losses_17560g7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ`
Њ ",Ђ)
"
tensor_0џџџџџџџџџ`
 
/__inference_FC1_preFlatten1_layer_call_fn_17554\7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ`
Њ "!
unknownџџџџџџџџџ`Є
"__inference__update_step_xla_16686~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`рфЉіаЯ?
Њ "
 
"__inference__update_step_xla_16691f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рцЉіаЯ?
Њ "
 
"__inference__update_step_xla_16696f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РчЕђаЯ?
Њ "
 
"__inference__update_step_xla_16701f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рэЕђаЯ?
Њ "
 Є
"__inference__update_step_xla_16706~xЂu
nЂk
!
gradient
<9	%Ђ"
њ

p
` VariableSpec 
`рКСпаЯ?
Њ "
 
"__inference__update_step_xla_16711f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ріЂѓаЯ?
Њ "
 
"__inference__update_step_xla_16716f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`паЯ?
Њ "
 
"__inference__update_step_xla_16721f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ЁпаЯ?
Њ "
 Є
"__inference__update_step_xla_16726~xЂu
nЂk
!
gradient 
<9	%Ђ"
њ 

p
` VariableSpec 
`лпаЯ?
Њ "
 
"__inference__update_step_xla_16731f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`рЎепаЯ?
Њ "
 
"__inference__update_step_xla_16736f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`рђпаЯ?
Њ "
 
"__inference__update_step_xla_16741f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
`љпаЯ?
Њ "
 Є
"__inference__update_step_xla_16746~xЂu
nЂk
!
gradient 0
<9	%Ђ"
њ 0

p
` VariableSpec 
`хпаЯ?
Њ "
 
"__inference__update_step_xla_16751f`Ђ]
VЂS

gradient0
0-	Ђ
њ0

p
` VariableSpec 
`рьѓаЯ?
Њ "
 
"__inference__update_step_xla_16756f`Ђ]
VЂS

gradient0
0-	Ђ
њ0

p
` VariableSpec 
`лпаЯ?
Њ "
 
"__inference__update_step_xla_16761f`Ђ]
VЂS

gradient0
0-	Ђ
њ0

p
` VariableSpec 
` УпаЯ?
Њ "
 Є
"__inference__update_step_xla_16766~xЂu
nЂk
!
gradient0@
<9	%Ђ"
њ0@

p
` VariableSpec 
`рпаЯ?
Њ "
 
"__inference__update_step_xla_16771f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рЎкпаЯ?
Њ "
 
"__inference__update_step_xla_16776f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`РипаЯ?
Њ "
 
"__inference__update_step_xla_16781f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
` зпаЯ?
Њ "
 Є
"__inference__update_step_xla_16786~xЂu
nЂk
!
gradient@`
<9	%Ђ"
њ@`

p
` VariableSpec 
` паЯ?
Њ "
 
"__inference__update_step_xla_16791f`Ђ]
VЂS

gradient`
0-	Ђ
њ`

p
` VariableSpec 
`рсОпаЯ?
Њ "
 
"__inference__update_step_xla_16796f`Ђ]
VЂS

gradient`
0-	Ђ
њ`

p
` VariableSpec 
`рќпаЯ?
Њ "
 
"__inference__update_step_xla_16801f`Ђ]
VЂS

gradient`
0-	Ђ
њ`

p
` VariableSpec 
`РФхпаЯ?
Њ "
 
"__inference__update_step_xla_16806nhЂe
^Ђ[

gradient`
41	Ђ
њ`

p
` VariableSpec 
`риџоаЯ?
Њ "
 
"__inference__update_step_xla_16811f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЩџоаЯ?
Њ "
 Я
 __inference__wrapped_model_14405Њ:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэю9Ђ6
/Ђ,
*'
input_1џџџџџџџџџ
Њ "1Њ.
,
flatten!
flattenџџџџџџџџџі
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_17549ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
1__inference_average_pooling2d_layer_call_fn_17544RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
B__inference_flatten_layer_call_and_return_conditional_losses_17600_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
'__inference_flatten_layer_call_fn_17594T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџє
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_17057ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_1_layer_call_fn_17052RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_17180ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_2_layer_call_fn_17175RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_17303ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_3_layer_call_fn_17298RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџє
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_17426ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ю
/__inference_max_pooling2d_4_layer_call_fn_17421RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџђ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_16934ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ь
-__inference_max_pooling2d_layer_call_fn_16929RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџђ
@__inference_model_layer_call_and_return_conditional_losses_15900­:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэюAЂ>
7Ђ4
*'
input_1џџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ђ
@__inference_model_layer_call_and_return_conditional_losses_16008­:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэюAЂ>
7Ђ4
*'
input_1џџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ё
@__inference_model_layer_call_and_return_conditional_losses_16408Ќ:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэю@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ё
@__inference_model_layer_call_and_return_conditional_losses_16681Ќ:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэю@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ь
%__inference_model_layer_call_fn_14921Ђ:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэюAЂ>
7Ђ4
*'
input_1џџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџЬ
%__inference_model_layer_call_fn_15792Ђ:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэюAЂ>
7Ђ4
*'
input_1џџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџЫ
%__inference_model_layer_call_fn_16174Ё:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэю@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџЫ
%__inference_model_layer_call_fn_16255Ё:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэю@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџЗ
B__inference_re_lu_1_layer_call_and_return_conditional_losses_17047q8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
'__inference_re_lu_1_layer_call_fn_17042f8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџЗ
B__inference_re_lu_2_layer_call_and_return_conditional_losses_17170q8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ 
 
'__inference_re_lu_2_layer_call_fn_17165f8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ 
Њ "*'
unknownџџџџџџџџџ З
B__inference_re_lu_3_layer_call_and_return_conditional_losses_17293q8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ0
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ0
 
'__inference_re_lu_3_layer_call_fn_17288f8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ0
Њ "*'
unknownџџџџџџџџџ0Е
B__inference_re_lu_4_layer_call_and_return_conditional_losses_17416o7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@@
 
'__inference_re_lu_4_layer_call_fn_17411d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@
Њ ")&
unknownџџџџџџџџџ@@Е
B__inference_re_lu_5_layer_call_and_return_conditional_losses_17539o7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ `
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ `
 
'__inference_re_lu_5_layer_call_fn_17534d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ `
Њ ")&
unknownџџџџџџџџџ `Е
@__inference_re_lu_layer_call_and_return_conditional_losses_16924q8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 
%__inference_re_lu_layer_call_fn_16919f8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџн
#__inference_signature_wrapper_16093Е:-.8976LMWXVUklvwutЉЊДЕГВШЩгдвбэюDЂA
Ђ 
:Њ7
5
input_1*'
input_1џџџџџџџџџ"1Њ.
,
flatten!
flattenџџџџџџџџџЉ
B__inference_softmax_layer_call_and_return_conditional_losses_17589c3Ђ0
)Ђ&
 
inputsџџџџџџџџџ

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
'__inference_softmax_layer_call_fn_17584X3Ђ0
)Ђ&
 
inputsџџџџџџџџџ

 
Њ "!
unknownџџџџџџџџџ