╘·
Ьы
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
╝
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
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
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
$
DisableCopyOnRead
resourceИ
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
В
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758е║

l

FC_2_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
FC_2_/bias
e
FC_2_/bias/Read/ReadVariableOpReadVariableOp
FC_2_/bias*
_output_shapes
:
*
dtype0
t
FC_2_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`
*
shared_nameFC_2_/kernel
m
 FC_2_/kernel/Read/ReadVariableOpReadVariableOpFC_2_/kernel*
_output_shapes

:`
*
dtype0
j
	FC0_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_name	FC0_/bias
c
FC0_/bias/Read/ReadVariableOpReadVariableOp	FC0_/bias*
_output_shapes
:`*
dtype0
r
FC0_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:``*
shared_nameFC0_/kernel
k
FC0_/kernel/Read/ReadVariableOpReadVariableOpFC0_/kernel*
_output_shapes

:``*
dtype0
l

CNN4_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_name
CNN4_/bias
e
CNN4_/bias/Read/ReadVariableOpReadVariableOp
CNN4_/bias*
_output_shapes
:`*
dtype0
|
CNN4_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*
shared_nameCNN4_/kernel
u
 CNN4_/kernel/Read/ReadVariableOpReadVariableOpCNN4_/kernel*&
_output_shapes
:@`*
dtype0
l

CNN3_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
CNN3_/bias
e
CNN3_/bias/Read/ReadVariableOpReadVariableOp
CNN3_/bias*
_output_shapes
:@*
dtype0
|
CNN3_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*
shared_nameCNN3_/kernel
u
 CNN3_/kernel/Read/ReadVariableOpReadVariableOpCNN3_/kernel*&
_output_shapes
:0@*
dtype0
l

CNN2_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name
CNN2_/bias
e
CNN2_/bias/Read/ReadVariableOpReadVariableOp
CNN2_/bias*
_output_shapes
:0*
dtype0
|
CNN2_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*
shared_nameCNN2_/kernel
u
 CNN2_/kernel/Read/ReadVariableOpReadVariableOpCNN2_/kernel*&
_output_shapes
: 0*
dtype0
l

CNN1_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
CNN1_/bias
e
CNN1_/bias/Read/ReadVariableOpReadVariableOp
CNN1_/bias*
_output_shapes
: *
dtype0
|
CNN1_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameCNN1_/kernel
u
 CNN1_/kernel/Read/ReadVariableOpReadVariableOpCNN1_/kernel*&
_output_shapes
: *
dtype0
l

CNN0_/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
CNN0_/bias
e
CNN0_/bias/Read/ReadVariableOpReadVariableOp
CNN0_/bias*
_output_shapes
:*
dtype0
|
CNN0_/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCNN0_/kernel
u
 CNN0_/kernel/Read/ReadVariableOpReadVariableOpCNN0_/kernel*&
_output_shapes
:*
dtype0
М
serving_default_input_1Placeholder*0
_output_shapes
:         А*
dtype0*%
shape:         А
Е
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1CNN0_/kernel
CNN0_/biasCNN1_/kernel
CNN1_/biasCNN2_/kernel
CNN2_/biasCNN3_/kernel
CNN3_/biasCNN4_/kernel
CNN4_/biasFC0_/kernel	FC0_/biasFC_2_/kernel
FC_2_/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_201308

NoOpNoOp
│s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*юr
valueфrBсr B┌r
л
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer_with_weights-6
layer-20
layer-21
layer-22
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

 _init_input_shape* 
╚
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op*
О
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
О
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
╚
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op*
О
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
О
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
╚
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
 S_jit_compiled_convolution_op*
О
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
О
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses* 
╚
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
 h_jit_compiled_convolution_op*
О
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
О
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses* 
╚
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

{kernel
|bias
 }_jit_compiled_convolution_op*
Т
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses* 
Ф
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses* 
Ф
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses* 
Ф
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses* 
о
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses
Ьkernel
	Эbias*
Ф
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses* 
о
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses
кkernel
	лbias*
Ф
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
░__call__
+▒&call_and_return_all_conditional_losses* 
Ф
▓	variables
│trainable_variables
┤regularization_losses
╡	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses* 
n
'0
(1
<2
=3
Q4
R5
f6
g7
{8
|9
Ь10
Э11
к12
л13*
n
'0
(1
<2
=3
Q4
R5
f6
g7
{8
|9
Ь10
Э11
к12
л13*
* 
╡
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
╜trace_0
╛trace_1
┐trace_2
└trace_3* 
:
┴trace_0
┬trace_1
├trace_2
─trace_3* 
* 

┼serving_default* 
* 

'0
(1*

'0
(1*
* 
Ш
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

╦trace_0* 

╠trace_0* 
\V
VARIABLE_VALUECNN0_/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN0_/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

╥trace_0* 

╙trace_0* 
* 
* 
* 
Ц
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

┘trace_0* 

┌trace_0* 

<0
=1*

<0
=1*
* 
Ш
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 
\V
VARIABLE_VALUECNN1_/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN1_/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

чtrace_0* 

шtrace_0* 
* 
* 
* 
Ц
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

юtrace_0* 

яtrace_0* 

Q0
R1*

Q0
R1*
* 
Ш
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

їtrace_0* 

Ўtrace_0* 
\V
VARIABLE_VALUECNN2_/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN2_/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

№trace_0* 

¤trace_0* 
* 
* 
* 
Ц
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses* 

Гtrace_0* 

Дtrace_0* 

f0
g1*

f0
g1*
* 
Ш
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
\V
VARIABLE_VALUECNN3_/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN3_/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

Сtrace_0* 

Тtrace_0* 
* 
* 
* 
Ц
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 

Шtrace_0* 

Щtrace_0* 

{0
|1*

{0
|1*
* 
Ш
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

Яtrace_0* 

аtrace_0* 
\V
VARIABLE_VALUECNN4_/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
CNN4_/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ъ
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
~	variables
trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses* 

жtrace_0* 

зtrace_0* 
* 
* 
* 
Ь
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 
* 
* 
* 
Ь
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses* 

┤trace_0* 

╡trace_0* 
* 
* 
* 
Ь
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses* 

╗trace_0* 

╝trace_0* 

Ь0
Э1*

Ь0
Э1*
* 
Ю
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses*

┬trace_0* 

├trace_0* 
[U
VARIABLE_VALUEFC0_/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	FC0_/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses* 

╔trace_0* 

╩trace_0* 

к0
л1*

к0
л1*
* 
Ю
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses*

╨trace_0* 

╤trace_0* 
\V
VARIABLE_VALUEFC_2_/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
FC_2_/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
м	variables
нtrainable_variables
оregularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses* 

╫trace_0* 

╪trace_0* 
* 
* 
* 
Ь
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
▌layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses* 

▐trace_0* 

▀trace_0* 
* 
▓
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
22*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
р
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameCNN0_/kernel
CNN0_/biasCNN1_/kernel
CNN1_/biasCNN2_/kernel
CNN2_/biasCNN3_/kernel
CNN3_/biasCNN4_/kernel
CNN4_/biasFC0_/kernel	FC0_/biasFC_2_/kernel
FC_2_/biasConst*
Tin
2*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_201892
█
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameCNN0_/kernel
CNN0_/biasCNN1_/kernel
CNN1_/biasCNN2_/kernel
CNN2_/biasCNN3_/kernel
CNN3_/biasCNN4_/kernel
CNN4_/biasFC0_/kernel	FC0_/biasFC_2_/kernel
FC_2_/bias*
Tin
2*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_201944╧а	
тn
╫
__inference__traced_save_201892
file_prefix=
#read_disablecopyonread_cnn0__kernel:1
#read_1_disablecopyonread_cnn0__bias:?
%read_2_disablecopyonread_cnn1__kernel: 1
#read_3_disablecopyonread_cnn1__bias: ?
%read_4_disablecopyonread_cnn2__kernel: 01
#read_5_disablecopyonread_cnn2__bias:0?
%read_6_disablecopyonread_cnn3__kernel:0@1
#read_7_disablecopyonread_cnn3__bias:@?
%read_8_disablecopyonread_cnn4__kernel:@`1
#read_9_disablecopyonread_cnn4__bias:`7
%read_10_disablecopyonread_fc0__kernel:``1
#read_11_disablecopyonread_fc0__bias:`8
&read_12_disablecopyonread_fc_2__kernel:`
2
$read_13_disablecopyonread_fc_2__bias:

savev2_const
identity_29ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_cnn0__kernel"/device:CPU:0*
_output_shapes
 з
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_cnn0__kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_cnn0__bias"/device:CPU:0*
_output_shapes
 Я
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_cnn0__bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_cnn1__kernel"/device:CPU:0*
_output_shapes
 н
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_cnn1__kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: w
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_cnn1__bias"/device:CPU:0*
_output_shapes
 Я
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_cnn1__bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_4/DisableCopyOnReadDisableCopyOnRead%read_4_disablecopyonread_cnn2__kernel"/device:CPU:0*
_output_shapes
 н
Read_4/ReadVariableOpReadVariableOp%read_4_disablecopyonread_cnn2__kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: 0*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: 0k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: 0w
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_cnn2__bias"/device:CPU:0*
_output_shapes
 Я
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_cnn2__bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:0y
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_cnn3__kernel"/device:CPU:0*
_output_shapes
 н
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_cnn3__kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:0@*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:0@m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:0@w
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_cnn3__bias"/device:CPU:0*
_output_shapes
 Я
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_cnn3__bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_8/DisableCopyOnReadDisableCopyOnRead%read_8_disablecopyonread_cnn4__kernel"/device:CPU:0*
_output_shapes
 н
Read_8/ReadVariableOpReadVariableOp%read_8_disablecopyonread_cnn4__kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@`*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@`m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:@`w
Read_9/DisableCopyOnReadDisableCopyOnRead#read_9_disablecopyonread_cnn4__bias"/device:CPU:0*
_output_shapes
 Я
Read_9/ReadVariableOpReadVariableOp#read_9_disablecopyonread_cnn4__bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:`z
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_fc0__kernel"/device:CPU:0*
_output_shapes
 з
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_fc0__kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:``*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:``e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:``x
Read_11/DisableCopyOnReadDisableCopyOnRead#read_11_disablecopyonread_fc0__bias"/device:CPU:0*
_output_shapes
 б
Read_11/ReadVariableOpReadVariableOp#read_11_disablecopyonread_fc0__bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:`*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:`a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:`{
Read_12/DisableCopyOnReadDisableCopyOnRead&read_12_disablecopyonread_fc_2__kernel"/device:CPU:0*
_output_shapes
 и
Read_12/ReadVariableOpReadVariableOp&read_12_disablecopyonread_fc_2__kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:`
*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:`
e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:`
y
Read_13/DisableCopyOnReadDisableCopyOnRead$read_13_disablecopyonread_fc_2__bias"/device:CPU:0*
_output_shapes
 в
Read_13/ReadVariableOpReadVariableOp$read_13_disablecopyonread_fc_2__bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
М
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╡
valueлBиB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЛ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B Х
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: й
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┘M
Й
A__inference_model_layer_call_and_return_conditional_losses_200965

inputs&
cnn0__200914:
cnn0__200916:&
cnn1__200921: 
cnn1__200923: &
cnn2__200928: 0
cnn2__200930:0&
cnn3__200935:0@
cnn3__200937:@&
cnn4__200942:@`
cnn4__200944:`
fc0__200951:``
fc0__200953:`
fc_2__200957:`

fc_2__200959:

identityИвCNN0_/StatefulPartitionedCallвCNN1_/StatefulPartitionedCallвCNN2_/StatefulPartitionedCallвCNN3_/StatefulPartitionedCallвCNN4_/StatefulPartitionedCallвFC0_/StatefulPartitionedCallвFC_2_/StatefulPartitionedCallЁ
CNN0_/StatefulPartitionedCallStatefulPartitionedCallinputscnn0__200914cnn0__200916*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN0__layer_call_and_return_conditional_losses_200680щ
MAX_POOL_0_/PartitionedCallPartitionedCall&CNN0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_200600у
CNN_REL0_/PartitionedCallPartitionedCall$MAX_POOL_0_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_200692Л
CNN1_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL0_/PartitionedCall:output:0cnn1__200921cnn1__200923*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN1__layer_call_and_return_conditional_losses_200704щ
MAX_POOL_1_/PartitionedCallPartitionedCall&CNN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_200612у
CNN_REL1_/PartitionedCallPartitionedCall$MAX_POOL_1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_200716Л
CNN2_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL1_/PartitionedCall:output:0cnn2__200928cnn2__200930*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN2__layer_call_and_return_conditional_losses_200728щ
MAX_POOL_2_/PartitionedCallPartitionedCall&CNN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_200624у
CNN_REL2_/PartitionedCallPartitionedCall$MAX_POOL_2_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_200740Л
CNN3_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL2_/PartitionedCall:output:0cnn3__200935cnn3__200937*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN3__layer_call_and_return_conditional_losses_200752щ
MAX_POOL_3_/PartitionedCallPartitionedCall&CNN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_200636у
CNN_REL3_/PartitionedCallPartitionedCall$MAX_POOL_3_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_200764Л
CNN4_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL3_/PartitionedCall:output:0cnn4__200942cnn4__200944*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN4__layer_call_and_return_conditional_losses_200776щ
MAX_POOL_4_/PartitionedCallPartitionedCall&CNN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_200648у
CNN_REL4_/PartitionedCallPartitionedCall$MAX_POOL_4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_200788┘
AVG1_/PartitionedCallPartitionedCall"CNN_REL4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_AVG1__layer_call_and_return_conditional_losses_200660═
FLT1_/PartitionedCallPartitionedCallAVG1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FLT1__layer_call_and_return_conditional_losses_200797√
FC0_/StatefulPartitionedCallStatefulPartitionedCallFLT1_/PartitionedCall:output:0fc0__200951fc0__200953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC0__layer_call_and_return_conditional_losses_200809▄
FC_RELU0_/PartitionedCallPartitionedCall%FC0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_200820Г
FC_2_/StatefulPartitionedCallStatefulPartitionedCall"FC_RELU0_/PartitionedCall:output:0fc_2__200957fc_2__200959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FC_2__layer_call_and_return_conditional_losses_200832┘
softmax/PartitionedCallPartitionedCall&FC_2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_200843╙
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_200851o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
е
NoOpNoOp^CNN0_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^FC0_/StatefulPartitionedCall^FC_2_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 2>
CNN0_/StatefulPartitionedCallCNN0_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2<
FC0_/StatefulPartitionedCallFC0_/StatefulPartitionedCall2>
FC_2_/StatefulPartitionedCallFC_2_/StatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_201646

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
─	
Є
A__inference_FC_2__layer_call_and_return_conditional_losses_200832

inputs0
matmul_readvariableop_resource:`
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
┘M
Й
A__inference_model_layer_call_and_return_conditional_losses_201052

inputs&
cnn0__201001:
cnn0__201003:&
cnn1__201008: 
cnn1__201010: &
cnn2__201015: 0
cnn2__201017:0&
cnn3__201022:0@
cnn3__201024:@&
cnn4__201029:@`
cnn4__201031:`
fc0__201038:``
fc0__201040:`
fc_2__201044:`

fc_2__201046:

identityИвCNN0_/StatefulPartitionedCallвCNN1_/StatefulPartitionedCallвCNN2_/StatefulPartitionedCallвCNN3_/StatefulPartitionedCallвCNN4_/StatefulPartitionedCallвFC0_/StatefulPartitionedCallвFC_2_/StatefulPartitionedCallЁ
CNN0_/StatefulPartitionedCallStatefulPartitionedCallinputscnn0__201001cnn0__201003*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN0__layer_call_and_return_conditional_losses_200680щ
MAX_POOL_0_/PartitionedCallPartitionedCall&CNN0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_200600у
CNN_REL0_/PartitionedCallPartitionedCall$MAX_POOL_0_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_200692Л
CNN1_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL0_/PartitionedCall:output:0cnn1__201008cnn1__201010*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN1__layer_call_and_return_conditional_losses_200704щ
MAX_POOL_1_/PartitionedCallPartitionedCall&CNN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_200612у
CNN_REL1_/PartitionedCallPartitionedCall$MAX_POOL_1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_200716Л
CNN2_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL1_/PartitionedCall:output:0cnn2__201015cnn2__201017*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN2__layer_call_and_return_conditional_losses_200728щ
MAX_POOL_2_/PartitionedCallPartitionedCall&CNN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_200624у
CNN_REL2_/PartitionedCallPartitionedCall$MAX_POOL_2_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_200740Л
CNN3_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL2_/PartitionedCall:output:0cnn3__201022cnn3__201024*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN3__layer_call_and_return_conditional_losses_200752щ
MAX_POOL_3_/PartitionedCallPartitionedCall&CNN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_200636у
CNN_REL3_/PartitionedCallPartitionedCall$MAX_POOL_3_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_200764Л
CNN4_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL3_/PartitionedCall:output:0cnn4__201029cnn4__201031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN4__layer_call_and_return_conditional_losses_200776щ
MAX_POOL_4_/PartitionedCallPartitionedCall&CNN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_200648у
CNN_REL4_/PartitionedCallPartitionedCall$MAX_POOL_4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_200788┘
AVG1_/PartitionedCallPartitionedCall"CNN_REL4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_AVG1__layer_call_and_return_conditional_losses_200660═
FLT1_/PartitionedCallPartitionedCallAVG1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FLT1__layer_call_and_return_conditional_losses_200797√
FC0_/StatefulPartitionedCallStatefulPartitionedCallFLT1_/PartitionedCall:output:0fc0__201038fc0__201040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC0__layer_call_and_return_conditional_losses_200809▄
FC_RELU0_/PartitionedCallPartitionedCall%FC0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_200820Г
FC_2_/StatefulPartitionedCallStatefulPartitionedCall"FC_RELU0_/PartitionedCall:output:0fc_2__201044fc_2__201046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FC_2__layer_call_and_return_conditional_losses_200832┘
softmax/PartitionedCallPartitionedCall&FC_2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_200843╙
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_200851o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
е
NoOpNoOp^CNN0_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^FC0_/StatefulPartitionedCall^FC_2_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 2>
CNN0_/StatefulPartitionedCallCNN0_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2<
FC0_/StatefulPartitionedCallFC0_/StatefulPartitionedCall2>
FC_2_/StatefulPartitionedCallFC_2_/StatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
┬
F
*__inference_CNN_REL0__layer_call_fn_201534

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_200692h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
│
H
,__inference_MAX_POOL_3__layer_call_fn_201641

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_200636Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
]
A__inference_AVG1__layer_call_and_return_conditional_losses_200660

inputs
identityл
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
├	
ё
@__inference_FC0__layer_call_and_return_conditional_losses_200809

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         `w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
┬
F
*__inference_CNN_REL4__layer_call_fn_201690

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_200788h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
│
H
,__inference_MAX_POOL_1__layer_call_fn_201563

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_200612Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
д

·
A__inference_CNN1__layer_call_and_return_conditional_losses_200704

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
│
H
,__inference_MAX_POOL_4__layer_call_fn_201680

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_200648Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▒>
╣
"__inference__traced_restore_201944
file_prefix7
assignvariableop_cnn0__kernel:+
assignvariableop_1_cnn0__bias:9
assignvariableop_2_cnn1__kernel: +
assignvariableop_3_cnn1__bias: 9
assignvariableop_4_cnn2__kernel: 0+
assignvariableop_5_cnn2__bias:09
assignvariableop_6_cnn3__kernel:0@+
assignvariableop_7_cnn3__bias:@9
assignvariableop_8_cnn4__kernel:@`+
assignvariableop_9_cnn4__bias:`1
assignvariableop_10_fc0__kernel:``+
assignvariableop_11_fc0__bias:`2
 assignvariableop_12_fc_2__kernel:`
,
assignvariableop_13_fc_2__bias:

identity_15ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9П
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╡
valueлBиB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHО
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B щ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOpAssignVariableOpassignvariableop_cnn0__kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_1AssignVariableOpassignvariableop_1_cnn0__biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_2AssignVariableOpassignvariableop_2_cnn1__kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_3AssignVariableOpassignvariableop_3_cnn1__biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_4AssignVariableOpassignvariableop_4_cnn2__kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_5AssignVariableOpassignvariableop_5_cnn2__biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_6AssignVariableOpassignvariableop_6_cnn3__kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_7AssignVariableOpassignvariableop_7_cnn3__biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_8AssignVariableOpassignvariableop_8_cnn4__kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_9AssignVariableOpassignvariableop_9_cnn4__biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_10AssignVariableOpassignvariableop_10_fc0__kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_11AssignVariableOpassignvariableop_11_fc0__biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_12AssignVariableOp assignvariableop_12_fc_2__kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_13AssignVariableOpassignvariableop_13_fc_2__biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: Ё
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ч
Ы
&__inference_CNN4__layer_call_fn_201665

inputs!
unknown:@`
	unknown_0:`
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN4__layer_call_and_return_conditional_losses_200776w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
▄M
К
A__inference_model_layer_call_and_return_conditional_losses_200908
input_1&
cnn0__200857:
cnn0__200859:&
cnn1__200864: 
cnn1__200866: &
cnn2__200871: 0
cnn2__200873:0&
cnn3__200878:0@
cnn3__200880:@&
cnn4__200885:@`
cnn4__200887:`
fc0__200894:``
fc0__200896:`
fc_2__200900:`

fc_2__200902:

identityИвCNN0_/StatefulPartitionedCallвCNN1_/StatefulPartitionedCallвCNN2_/StatefulPartitionedCallвCNN3_/StatefulPartitionedCallвCNN4_/StatefulPartitionedCallвFC0_/StatefulPartitionedCallвFC_2_/StatefulPartitionedCallё
CNN0_/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn0__200857cnn0__200859*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN0__layer_call_and_return_conditional_losses_200680щ
MAX_POOL_0_/PartitionedCallPartitionedCall&CNN0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_200600у
CNN_REL0_/PartitionedCallPartitionedCall$MAX_POOL_0_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_200692Л
CNN1_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL0_/PartitionedCall:output:0cnn1__200864cnn1__200866*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN1__layer_call_and_return_conditional_losses_200704щ
MAX_POOL_1_/PartitionedCallPartitionedCall&CNN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_200612у
CNN_REL1_/PartitionedCallPartitionedCall$MAX_POOL_1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_200716Л
CNN2_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL1_/PartitionedCall:output:0cnn2__200871cnn2__200873*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN2__layer_call_and_return_conditional_losses_200728щ
MAX_POOL_2_/PartitionedCallPartitionedCall&CNN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_200624у
CNN_REL2_/PartitionedCallPartitionedCall$MAX_POOL_2_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_200740Л
CNN3_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL2_/PartitionedCall:output:0cnn3__200878cnn3__200880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN3__layer_call_and_return_conditional_losses_200752щ
MAX_POOL_3_/PartitionedCallPartitionedCall&CNN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_200636у
CNN_REL3_/PartitionedCallPartitionedCall$MAX_POOL_3_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_200764Л
CNN4_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL3_/PartitionedCall:output:0cnn4__200885cnn4__200887*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN4__layer_call_and_return_conditional_losses_200776щ
MAX_POOL_4_/PartitionedCallPartitionedCall&CNN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_200648у
CNN_REL4_/PartitionedCallPartitionedCall$MAX_POOL_4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_200788┘
AVG1_/PartitionedCallPartitionedCall"CNN_REL4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_AVG1__layer_call_and_return_conditional_losses_200660═
FLT1_/PartitionedCallPartitionedCallAVG1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FLT1__layer_call_and_return_conditional_losses_200797√
FC0_/StatefulPartitionedCallStatefulPartitionedCallFLT1_/PartitionedCall:output:0fc0__200894fc0__200896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC0__layer_call_and_return_conditional_losses_200809▄
FC_RELU0_/PartitionedCallPartitionedCall%FC0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_200820Г
FC_2_/StatefulPartitionedCallStatefulPartitionedCall"FC_RELU0_/PartitionedCall:output:0fc_2__200900fc_2__200902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FC_2__layer_call_and_return_conditional_losses_200832┘
softmax/PartitionedCallPartitionedCall&FC_2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_200843╙
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_200851o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
е
NoOpNoOp^CNN0_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^FC0_/StatefulPartitionedCall^FC_2_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 2>
CNN0_/StatefulPartitionedCallCNN0_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2<
FC0_/StatefulPartitionedCallFC0_/StatefulPartitionedCall2>
FC_2_/StatefulPartitionedCallFC_2_/StatefulPartitionedCall:Y U
0
_output_shapes
:         А
!
_user_specified_name	input_1
д

·
A__inference_CNN4__layer_call_and_return_conditional_losses_201675

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `*
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
:          `g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:          `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
│
H
,__inference_MAX_POOL_0__layer_call_fn_201524

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_200600Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_201656

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:          @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          @:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
д

·
A__inference_CNN2__layer_call_and_return_conditional_losses_201597

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
▄M
К
A__inference_model_layer_call_and_return_conditional_losses_200854
input_1&
cnn0__200681:
cnn0__200683:&
cnn1__200705: 
cnn1__200707: &
cnn2__200729: 0
cnn2__200731:0&
cnn3__200753:0@
cnn3__200755:@&
cnn4__200777:@`
cnn4__200779:`
fc0__200810:``
fc0__200812:`
fc_2__200833:`

fc_2__200835:

identityИвCNN0_/StatefulPartitionedCallвCNN1_/StatefulPartitionedCallвCNN2_/StatefulPartitionedCallвCNN3_/StatefulPartitionedCallвCNN4_/StatefulPartitionedCallвFC0_/StatefulPartitionedCallвFC_2_/StatefulPartitionedCallё
CNN0_/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn0__200681cnn0__200683*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN0__layer_call_and_return_conditional_losses_200680щ
MAX_POOL_0_/PartitionedCallPartitionedCall&CNN0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_200600у
CNN_REL0_/PartitionedCallPartitionedCall$MAX_POOL_0_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_200692Л
CNN1_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL0_/PartitionedCall:output:0cnn1__200705cnn1__200707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN1__layer_call_and_return_conditional_losses_200704щ
MAX_POOL_1_/PartitionedCallPartitionedCall&CNN1_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_200612у
CNN_REL1_/PartitionedCallPartitionedCall$MAX_POOL_1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_200716Л
CNN2_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL1_/PartitionedCall:output:0cnn2__200729cnn2__200731*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN2__layer_call_and_return_conditional_losses_200728щ
MAX_POOL_2_/PartitionedCallPartitionedCall&CNN2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_200624у
CNN_REL2_/PartitionedCallPartitionedCall$MAX_POOL_2_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_200740Л
CNN3_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL2_/PartitionedCall:output:0cnn3__200753cnn3__200755*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN3__layer_call_and_return_conditional_losses_200752щ
MAX_POOL_3_/PartitionedCallPartitionedCall&CNN3_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_200636у
CNN_REL3_/PartitionedCallPartitionedCall$MAX_POOL_3_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_200764Л
CNN4_/StatefulPartitionedCallStatefulPartitionedCall"CNN_REL3_/PartitionedCall:output:0cnn4__200777cnn4__200779*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN4__layer_call_and_return_conditional_losses_200776щ
MAX_POOL_4_/PartitionedCallPartitionedCall&CNN4_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_200648у
CNN_REL4_/PartitionedCallPartitionedCall$MAX_POOL_4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_200788┘
AVG1_/PartitionedCallPartitionedCall"CNN_REL4_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_AVG1__layer_call_and_return_conditional_losses_200660═
FLT1_/PartitionedCallPartitionedCallAVG1_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FLT1__layer_call_and_return_conditional_losses_200797√
FC0_/StatefulPartitionedCallStatefulPartitionedCallFLT1_/PartitionedCall:output:0fc0__200810fc0__200812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC0__layer_call_and_return_conditional_losses_200809▄
FC_RELU0_/PartitionedCallPartitionedCall%FC0_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_200820Г
FC_2_/StatefulPartitionedCallStatefulPartitionedCall"FC_RELU0_/PartitionedCall:output:0fc_2__200833fc_2__200835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FC_2__layer_call_and_return_conditional_losses_200832┘
softmax/PartitionedCallPartitionedCall&FC_2_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_200843╙
flatten/PartitionedCallPartitionedCall softmax/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_200851o
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
е
NoOpNoOp^CNN0_/StatefulPartitionedCall^CNN1_/StatefulPartitionedCall^CNN2_/StatefulPartitionedCall^CNN3_/StatefulPartitionedCall^CNN4_/StatefulPartitionedCall^FC0_/StatefulPartitionedCall^FC_2_/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 2>
CNN0_/StatefulPartitionedCallCNN0_/StatefulPartitionedCall2>
CNN1_/StatefulPartitionedCallCNN1_/StatefulPartitionedCall2>
CNN2_/StatefulPartitionedCallCNN2_/StatefulPartitionedCall2>
CNN3_/StatefulPartitionedCallCNN3_/StatefulPartitionedCall2>
CNN4_/StatefulPartitionedCallCNN4_/StatefulPartitionedCall2<
FC0_/StatefulPartitionedCallFC0_/StatefulPartitionedCall2>
FC_2_/StatefulPartitionedCallFC_2_/StatefulPartitionedCall:Y U
0
_output_shapes
:         А
!
_user_specified_name	input_1
П
c
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_200624

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ч
Ы
&__inference_CNN2__layer_call_fn_201587

inputs!
unknown: 0
	unknown_0:0
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN2__layer_call_and_return_conditional_losses_200728w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_200612

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
·
Г
&__inference_model_layer_call_fn_201083
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@`
	unknown_8:`
	unknown_9:``

unknown_10:`

unknown_11:`


unknown_12:

identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_201052o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         А
!
_user_specified_name	input_1
щ
a
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_200788

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
з
B
&__inference_AVG1__layer_call_fn_201700

inputs
identity╥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_AVG1__layer_call_and_return_conditional_losses_200660Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_200764

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:          @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          @:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
д

·
A__inference_CNN3__layer_call_and_return_conditional_losses_201636

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
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
:          @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:          @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          0
 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_200600

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_201539

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╔
a
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_200820

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:         `Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         `:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_201568

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╔
a
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_201745

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:         `Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         `:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
├	
ё
@__inference_FC0__layer_call_and_return_conditional_losses_201735

inputs0
matmul_readvariableop_resource:``-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:``*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         `w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
ў
В
&__inference_model_layer_call_fn_201374

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@`
	unknown_8:`
	unknown_9:``

unknown_10:`

unknown_11:`


unknown_12:

identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_201052o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
·
Г
&__inference_model_layer_call_fn_200996
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@`
	unknown_8:`
	unknown_9:``

unknown_10:`

unknown_11:`


unknown_12:

identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_200965o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         А
!
_user_specified_name	input_1
П
c
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_200648

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
├P
│
!__inference__wrapped_model_200594
input_1D
*model_cnn0__conv2d_readvariableop_resource:9
+model_cnn0__biasadd_readvariableop_resource:D
*model_cnn1__conv2d_readvariableop_resource: 9
+model_cnn1__biasadd_readvariableop_resource: D
*model_cnn2__conv2d_readvariableop_resource: 09
+model_cnn2__biasadd_readvariableop_resource:0D
*model_cnn3__conv2d_readvariableop_resource:0@9
+model_cnn3__biasadd_readvariableop_resource:@D
*model_cnn4__conv2d_readvariableop_resource:@`9
+model_cnn4__biasadd_readvariableop_resource:`;
)model_fc0__matmul_readvariableop_resource:``8
*model_fc0__biasadd_readvariableop_resource:`<
*model_fc_2__matmul_readvariableop_resource:`
9
+model_fc_2__biasadd_readvariableop_resource:

identityИв"model/CNN0_/BiasAdd/ReadVariableOpв!model/CNN0_/Conv2D/ReadVariableOpв"model/CNN1_/BiasAdd/ReadVariableOpв!model/CNN1_/Conv2D/ReadVariableOpв"model/CNN2_/BiasAdd/ReadVariableOpв!model/CNN2_/Conv2D/ReadVariableOpв"model/CNN3_/BiasAdd/ReadVariableOpв!model/CNN3_/Conv2D/ReadVariableOpв"model/CNN4_/BiasAdd/ReadVariableOpв!model/CNN4_/Conv2D/ReadVariableOpв!model/FC0_/BiasAdd/ReadVariableOpв model/FC0_/MatMul/ReadVariableOpв"model/FC_2_/BiasAdd/ReadVariableOpв!model/FC_2_/MatMul/ReadVariableOpФ
!model/CNN0_/Conv2D/ReadVariableOpReadVariableOp*model_cnn0__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0│
model/CNN0_/Conv2DConv2Dinput_1)model/CNN0_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
К
"model/CNN0_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn0__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0в
model/CNN0_/BiasAddBiasAddmodel/CNN0_/Conv2D:output:0*model/CNN0_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Ап
model/MAX_POOL_0_/MaxPoolMaxPoolmodel/CNN0_/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
z
model/CNN_REL0_/ReluRelu"model/MAX_POOL_0_/MaxPool:output:0*
T0*/
_output_shapes
:         @Ф
!model/CNN1_/Conv2D/ReadVariableOpReadVariableOp*model_cnn1__conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0═
model/CNN1_/Conv2DConv2D"model/CNN_REL0_/Relu:activations:0)model/CNN1_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ *
paddingSAME*
strides
К
"model/CNN1_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn1__biasadd_readvariableop_resource*
_output_shapes
: *
dtype0б
model/CNN1_/BiasAddBiasAddmodel/CNN1_/Conv2D:output:0*model/CNN1_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ п
model/MAX_POOL_1_/MaxPoolMaxPoolmodel/CNN1_/BiasAdd:output:0*/
_output_shapes
:         @ *
ksize
*
paddingVALID*
strides
z
model/CNN_REL1_/ReluRelu"model/MAX_POOL_1_/MaxPool:output:0*
T0*/
_output_shapes
:         @ Ф
!model/CNN2_/Conv2D/ReadVariableOpReadVariableOp*model_cnn2__conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0═
model/CNN2_/Conv2DConv2D"model/CNN_REL1_/Relu:activations:0)model/CNN2_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0*
paddingSAME*
strides
К
"model/CNN2_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn2__biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0б
model/CNN2_/BiasAddBiasAddmodel/CNN2_/Conv2D:output:0*model/CNN2_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0п
model/MAX_POOL_2_/MaxPoolMaxPoolmodel/CNN2_/BiasAdd:output:0*/
_output_shapes
:          0*
ksize
*
paddingVALID*
strides
z
model/CNN_REL2_/ReluRelu"model/MAX_POOL_2_/MaxPool:output:0*
T0*/
_output_shapes
:          0Ф
!model/CNN3_/Conv2D/ReadVariableOpReadVariableOp*model_cnn3__conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0═
model/CNN3_/Conv2DConv2D"model/CNN_REL2_/Relu:activations:0)model/CNN3_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
К
"model/CNN3_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn3__biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0б
model/CNN3_/BiasAddBiasAddmodel/CNN3_/Conv2D:output:0*model/CNN3_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @п
model/MAX_POOL_3_/MaxPoolMaxPoolmodel/CNN3_/BiasAdd:output:0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
z
model/CNN_REL3_/ReluRelu"model/MAX_POOL_3_/MaxPool:output:0*
T0*/
_output_shapes
:          @Ф
!model/CNN4_/Conv2D/ReadVariableOpReadVariableOp*model_cnn4__conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0═
model/CNN4_/Conv2DConv2D"model/CNN_REL3_/Relu:activations:0)model/CNN4_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `*
paddingSAME*
strides
К
"model/CNN4_/BiasAdd/ReadVariableOpReadVariableOp+model_cnn4__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0б
model/CNN4_/BiasAddBiasAddmodel/CNN4_/Conv2D:output:0*model/CNN4_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `п
model/MAX_POOL_4_/MaxPoolMaxPoolmodel/CNN4_/BiasAdd:output:0*/
_output_shapes
:         `*
ksize
*
paddingVALID*
strides
z
model/CNN_REL4_/ReluRelu"model/MAX_POOL_4_/MaxPool:output:0*
T0*/
_output_shapes
:         `╕
model/AVG1_/AvgPoolAvgPool"model/CNN_REL4_/Relu:activations:0*
T0*/
_output_shapes
:         `*
ksize
*
paddingVALID*
strides
b
model/FLT1_/ConstConst*
_output_shapes
:*
dtype0*
valueB"    `   К
model/FLT1_/ReshapeReshapemodel/AVG1_/AvgPool:output:0model/FLT1_/Const:output:0*
T0*'
_output_shapes
:         `К
 model/FC0_/MatMul/ReadVariableOpReadVariableOp)model_fc0__matmul_readvariableop_resource*
_output_shapes

:``*
dtype0Х
model/FC0_/MatMulMatMulmodel/FLT1_/Reshape:output:0(model/FC0_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `И
!model/FC0_/BiasAdd/ReadVariableOpReadVariableOp*model_fc0__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ч
model/FC0_/BiasAddBiasAddmodel/FC0_/MatMul:product:0)model/FC0_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `k
model/FC_RELU0_/ReluRelumodel/FC0_/BiasAdd:output:0*
T0*'
_output_shapes
:         `М
!model/FC_2_/MatMul/ReadVariableOpReadVariableOp*model_fc_2__matmul_readvariableop_resource*
_output_shapes

:`
*
dtype0Э
model/FC_2_/MatMulMatMul"model/FC_RELU0_/Relu:activations:0)model/FC_2_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
К
"model/FC_2_/BiasAdd/ReadVariableOpReadVariableOp+model_fc_2__biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ъ
model/FC_2_/BiasAddBiasAddmodel/FC_2_/MatMul:product:0*model/FC_2_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
p
model/softmax/SoftmaxSoftmaxmodel/FC_2_/BiasAdd:output:0*
T0*'
_output_shapes
:         
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   С
model/flatten/ReshapeReshapemodel/softmax/Softmax:softmax:0model/flatten/Const:output:0*
T0*'
_output_shapes
:         
m
IdentityIdentitymodel/flatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:         
├
NoOpNoOp#^model/CNN0_/BiasAdd/ReadVariableOp"^model/CNN0_/Conv2D/ReadVariableOp#^model/CNN1_/BiasAdd/ReadVariableOp"^model/CNN1_/Conv2D/ReadVariableOp#^model/CNN2_/BiasAdd/ReadVariableOp"^model/CNN2_/Conv2D/ReadVariableOp#^model/CNN3_/BiasAdd/ReadVariableOp"^model/CNN3_/Conv2D/ReadVariableOp#^model/CNN4_/BiasAdd/ReadVariableOp"^model/CNN4_/Conv2D/ReadVariableOp"^model/FC0_/BiasAdd/ReadVariableOp!^model/FC0_/MatMul/ReadVariableOp#^model/FC_2_/BiasAdd/ReadVariableOp"^model/FC_2_/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 2H
"model/CNN0_/BiasAdd/ReadVariableOp"model/CNN0_/BiasAdd/ReadVariableOp2F
!model/CNN0_/Conv2D/ReadVariableOp!model/CNN0_/Conv2D/ReadVariableOp2H
"model/CNN1_/BiasAdd/ReadVariableOp"model/CNN1_/BiasAdd/ReadVariableOp2F
!model/CNN1_/Conv2D/ReadVariableOp!model/CNN1_/Conv2D/ReadVariableOp2H
"model/CNN2_/BiasAdd/ReadVariableOp"model/CNN2_/BiasAdd/ReadVariableOp2F
!model/CNN2_/Conv2D/ReadVariableOp!model/CNN2_/Conv2D/ReadVariableOp2H
"model/CNN3_/BiasAdd/ReadVariableOp"model/CNN3_/BiasAdd/ReadVariableOp2F
!model/CNN3_/Conv2D/ReadVariableOp!model/CNN3_/Conv2D/ReadVariableOp2H
"model/CNN4_/BiasAdd/ReadVariableOp"model/CNN4_/BiasAdd/ReadVariableOp2F
!model/CNN4_/Conv2D/ReadVariableOp!model/CNN4_/Conv2D/ReadVariableOp2F
!model/FC0_/BiasAdd/ReadVariableOp!model/FC0_/BiasAdd/ReadVariableOp2D
 model/FC0_/MatMul/ReadVariableOp model/FC0_/MatMul/ReadVariableOp2H
"model/FC_2_/BiasAdd/ReadVariableOp"model/FC_2_/BiasAdd/ReadVariableOp2F
!model/FC_2_/MatMul/ReadVariableOp!model/FC_2_/MatMul/ReadVariableOp:Y U
0
_output_shapes
:         А
!
_user_specified_name	input_1
╪
Б
$__inference_signature_wrapper_201308
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@`
	unknown_8:`
	unknown_9:``

unknown_10:`

unknown_11:`


unknown_12:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_200594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         А
!
_user_specified_name	input_1
щ
a
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_200716

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         @ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @ :W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
ч
Ы
&__inference_CNN1__layer_call_fn_201548

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN1__layer_call_and_return_conditional_losses_200704w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
д

·
A__inference_CNN4__layer_call_and_return_conditional_losses_200776

inputs8
conv2d_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `*
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
:          `g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:          `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
в
F
*__inference_FC_RELU0__layer_call_fn_201740

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_200820`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         `:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
┐
У
&__inference_FC_2__layer_call_fn_201754

inputs
unknown:`

	unknown_0:

identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FC_2__layer_call_and_return_conditional_losses_200832o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_201607

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╠
_
C__inference_softmax_layer_call_and_return_conditional_losses_200843

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
д

·
A__inference_CNN1__layer_call_and_return_conditional_losses_201558

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_200636

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_200740

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:          0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          0:W S
/
_output_shapes
:          0
 
_user_specified_nameinputs
┬H
к

A__inference_model_layer_call_and_return_conditional_losses_201437

inputs>
$cnn0__conv2d_readvariableop_resource:3
%cnn0__biasadd_readvariableop_resource:>
$cnn1__conv2d_readvariableop_resource: 3
%cnn1__biasadd_readvariableop_resource: >
$cnn2__conv2d_readvariableop_resource: 03
%cnn2__biasadd_readvariableop_resource:0>
$cnn3__conv2d_readvariableop_resource:0@3
%cnn3__biasadd_readvariableop_resource:@>
$cnn4__conv2d_readvariableop_resource:@`3
%cnn4__biasadd_readvariableop_resource:`5
#fc0__matmul_readvariableop_resource:``2
$fc0__biasadd_readvariableop_resource:`6
$fc_2__matmul_readvariableop_resource:`
3
%fc_2__biasadd_readvariableop_resource:

identityИвCNN0_/BiasAdd/ReadVariableOpвCNN0_/Conv2D/ReadVariableOpвCNN1_/BiasAdd/ReadVariableOpвCNN1_/Conv2D/ReadVariableOpвCNN2_/BiasAdd/ReadVariableOpвCNN2_/Conv2D/ReadVariableOpвCNN3_/BiasAdd/ReadVariableOpвCNN3_/Conv2D/ReadVariableOpвCNN4_/BiasAdd/ReadVariableOpвCNN4_/Conv2D/ReadVariableOpвFC0_/BiasAdd/ReadVariableOpвFC0_/MatMul/ReadVariableOpвFC_2_/BiasAdd/ReadVariableOpвFC_2_/MatMul/ReadVariableOpИ
CNN0_/Conv2D/ReadVariableOpReadVariableOp$cnn0__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ж
CNN0_/Conv2DConv2Dinputs#CNN0_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
~
CNN0_/BiasAdd/ReadVariableOpReadVariableOp%cnn0__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Р
CNN0_/BiasAddBiasAddCNN0_/Conv2D:output:0$CNN0_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аг
MAX_POOL_0_/MaxPoolMaxPoolCNN0_/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
n
CNN_REL0_/ReluReluMAX_POOL_0_/MaxPool:output:0*
T0*/
_output_shapes
:         @И
CNN1_/Conv2D/ReadVariableOpReadVariableOp$cnn1__conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╗
CNN1_/Conv2DConv2DCNN_REL0_/Relu:activations:0#CNN1_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ *
paddingSAME*
strides
~
CNN1_/BiasAdd/ReadVariableOpReadVariableOp%cnn1__biasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
CNN1_/BiasAddBiasAddCNN1_/Conv2D:output:0$CNN1_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ г
MAX_POOL_1_/MaxPoolMaxPoolCNN1_/BiasAdd:output:0*/
_output_shapes
:         @ *
ksize
*
paddingVALID*
strides
n
CNN_REL1_/ReluReluMAX_POOL_1_/MaxPool:output:0*
T0*/
_output_shapes
:         @ И
CNN2_/Conv2D/ReadVariableOpReadVariableOp$cnn2__conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0╗
CNN2_/Conv2DConv2DCNN_REL1_/Relu:activations:0#CNN2_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0*
paddingSAME*
strides
~
CNN2_/BiasAdd/ReadVariableOpReadVariableOp%cnn2__biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0П
CNN2_/BiasAddBiasAddCNN2_/Conv2D:output:0$CNN2_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0г
MAX_POOL_2_/MaxPoolMaxPoolCNN2_/BiasAdd:output:0*/
_output_shapes
:          0*
ksize
*
paddingVALID*
strides
n
CNN_REL2_/ReluReluMAX_POOL_2_/MaxPool:output:0*
T0*/
_output_shapes
:          0И
CNN3_/Conv2D/ReadVariableOpReadVariableOp$cnn3__conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0╗
CNN3_/Conv2DConv2DCNN_REL2_/Relu:activations:0#CNN3_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
~
CNN3_/BiasAdd/ReadVariableOpReadVariableOp%cnn3__biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
CNN3_/BiasAddBiasAddCNN3_/Conv2D:output:0$CNN3_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @г
MAX_POOL_3_/MaxPoolMaxPoolCNN3_/BiasAdd:output:0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
n
CNN_REL3_/ReluReluMAX_POOL_3_/MaxPool:output:0*
T0*/
_output_shapes
:          @И
CNN4_/Conv2D/ReadVariableOpReadVariableOp$cnn4__conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0╗
CNN4_/Conv2DConv2DCNN_REL3_/Relu:activations:0#CNN4_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `*
paddingSAME*
strides
~
CNN4_/BiasAdd/ReadVariableOpReadVariableOp%cnn4__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0П
CNN4_/BiasAddBiasAddCNN4_/Conv2D:output:0$CNN4_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `г
MAX_POOL_4_/MaxPoolMaxPoolCNN4_/BiasAdd:output:0*/
_output_shapes
:         `*
ksize
*
paddingVALID*
strides
n
CNN_REL4_/ReluReluMAX_POOL_4_/MaxPool:output:0*
T0*/
_output_shapes
:         `м
AVG1_/AvgPoolAvgPoolCNN_REL4_/Relu:activations:0*
T0*/
_output_shapes
:         `*
ksize
*
paddingVALID*
strides
\
FLT1_/ConstConst*
_output_shapes
:*
dtype0*
valueB"    `   x
FLT1_/ReshapeReshapeAVG1_/AvgPool:output:0FLT1_/Const:output:0*
T0*'
_output_shapes
:         `~
FC0_/MatMul/ReadVariableOpReadVariableOp#fc0__matmul_readvariableop_resource*
_output_shapes

:``*
dtype0Г
FC0_/MatMulMatMulFLT1_/Reshape:output:0"FC0_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `|
FC0_/BiasAdd/ReadVariableOpReadVariableOp$fc0__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
FC0_/BiasAddBiasAddFC0_/MatMul:product:0#FC0_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `_
FC_RELU0_/ReluReluFC0_/BiasAdd:output:0*
T0*'
_output_shapes
:         `А
FC_2_/MatMul/ReadVariableOpReadVariableOp$fc_2__matmul_readvariableop_resource*
_output_shapes

:`
*
dtype0Л
FC_2_/MatMulMatMulFC_RELU0_/Relu:activations:0#FC_2_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
~
FC_2_/BiasAdd/ReadVariableOpReadVariableOp%fc_2__biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0И
FC_2_/BiasAddBiasAddFC_2_/MatMul:product:0$FC_2_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
softmax/SoftmaxSoftmaxFC_2_/BiasAdd:output:0*
T0*'
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   
flatten/ReshapeReshapesoftmax/Softmax:softmax:0flatten/Const:output:0*
T0*'
_output_shapes
:         
g
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:         
я
NoOpNoOp^CNN0_/BiasAdd/ReadVariableOp^CNN0_/Conv2D/ReadVariableOp^CNN1_/BiasAdd/ReadVariableOp^CNN1_/Conv2D/ReadVariableOp^CNN2_/BiasAdd/ReadVariableOp^CNN2_/Conv2D/ReadVariableOp^CNN3_/BiasAdd/ReadVariableOp^CNN3_/Conv2D/ReadVariableOp^CNN4_/BiasAdd/ReadVariableOp^CNN4_/Conv2D/ReadVariableOp^FC0_/BiasAdd/ReadVariableOp^FC0_/MatMul/ReadVariableOp^FC_2_/BiasAdd/ReadVariableOp^FC_2_/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 2<
CNN0_/BiasAdd/ReadVariableOpCNN0_/BiasAdd/ReadVariableOp2:
CNN0_/Conv2D/ReadVariableOpCNN0_/Conv2D/ReadVariableOp2<
CNN1_/BiasAdd/ReadVariableOpCNN1_/BiasAdd/ReadVariableOp2:
CNN1_/Conv2D/ReadVariableOpCNN1_/Conv2D/ReadVariableOp2<
CNN2_/BiasAdd/ReadVariableOpCNN2_/BiasAdd/ReadVariableOp2:
CNN2_/Conv2D/ReadVariableOpCNN2_/Conv2D/ReadVariableOp2<
CNN3_/BiasAdd/ReadVariableOpCNN3_/BiasAdd/ReadVariableOp2:
CNN3_/Conv2D/ReadVariableOpCNN3_/Conv2D/ReadVariableOp2<
CNN4_/BiasAdd/ReadVariableOpCNN4_/BiasAdd/ReadVariableOp2:
CNN4_/Conv2D/ReadVariableOpCNN4_/Conv2D/ReadVariableOp2:
FC0_/BiasAdd/ReadVariableOpFC0_/BiasAdd/ReadVariableOp28
FC0_/MatMul/ReadVariableOpFC0_/MatMul/ReadVariableOp2<
FC_2_/BiasAdd/ReadVariableOpFC_2_/BiasAdd/ReadVariableOp2:
FC_2_/MatMul/ReadVariableOpFC_2_/MatMul/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
к
B
&__inference_FLT1__layer_call_fn_201710

inputs
identityп
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_FLT1__layer_call_and_return_conditional_losses_200797`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_200692

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
д

·
A__inference_CNN2__layer_call_and_return_conditional_losses_200728

inputs8
conv2d_readvariableop_resource: 0-
biasadd_readvariableop_resource:0
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
─	
Є
A__inference_FC_2__layer_call_and_return_conditional_losses_201764

inputs0
matmul_readvariableop_resource:`
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
╜
Т
%__inference_FC0__layer_call_fn_201725

inputs
unknown:``
	unknown_0:`
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC0__layer_call_and_return_conditional_losses_200809o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         `
 
_user_specified_nameinputs
д

·
A__inference_CNN3__layer_call_and_return_conditional_losses_200752

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
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
:          @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:          @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          0
 
_user_specified_nameinputs
│
_
C__inference_flatten_layer_call_and_return_conditional_losses_200851

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
ы
Ы
&__inference_CNN0__layer_call_fn_201509

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN0__layer_call_and_return_conditional_losses_200680x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ю
D
(__inference_flatten_layer_call_fn_201779

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_200851`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_201685

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╠
_
C__inference_softmax_layer_call_and_return_conditional_losses_201774

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_201617

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:          0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:          0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          0:W S
/
_output_shapes
:          0
 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_201578

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         @ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @ :W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
┬
F
*__inference_CNN_REL1__layer_call_fn_201573

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_200716h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @ :W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
┴
]
A__inference_FLT1__layer_call_and_return_conditional_losses_201716

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    `   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         `X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
й

·
A__inference_CNN0__layer_call_and_return_conditional_losses_201519

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
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
:         Аh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
]
A__inference_AVG1__layer_call_and_return_conditional_losses_201705

inputs
identityл
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
│
H
,__inference_MAX_POOL_2__layer_call_fn_201602

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_200624Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┴
]
A__inference_FLT1__layer_call_and_return_conditional_losses_200797

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    `   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         `X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
┬
F
*__inference_CNN_REL2__layer_call_fn_201612

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_200740h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          0:W S
/
_output_shapes
:          0
 
_user_specified_nameinputs
│
_
C__inference_flatten_layer_call_and_return_conditional_losses_201785

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Ю
D
(__inference_softmax_layer_call_fn_201769

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_200843`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         
:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
┬
F
*__inference_CNN_REL3__layer_call_fn_201651

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_200764h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          @:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
П
c
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_201529

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┬H
к

A__inference_model_layer_call_and_return_conditional_losses_201500

inputs>
$cnn0__conv2d_readvariableop_resource:3
%cnn0__biasadd_readvariableop_resource:>
$cnn1__conv2d_readvariableop_resource: 3
%cnn1__biasadd_readvariableop_resource: >
$cnn2__conv2d_readvariableop_resource: 03
%cnn2__biasadd_readvariableop_resource:0>
$cnn3__conv2d_readvariableop_resource:0@3
%cnn3__biasadd_readvariableop_resource:@>
$cnn4__conv2d_readvariableop_resource:@`3
%cnn4__biasadd_readvariableop_resource:`5
#fc0__matmul_readvariableop_resource:``2
$fc0__biasadd_readvariableop_resource:`6
$fc_2__matmul_readvariableop_resource:`
3
%fc_2__biasadd_readvariableop_resource:

identityИвCNN0_/BiasAdd/ReadVariableOpвCNN0_/Conv2D/ReadVariableOpвCNN1_/BiasAdd/ReadVariableOpвCNN1_/Conv2D/ReadVariableOpвCNN2_/BiasAdd/ReadVariableOpвCNN2_/Conv2D/ReadVariableOpвCNN3_/BiasAdd/ReadVariableOpвCNN3_/Conv2D/ReadVariableOpвCNN4_/BiasAdd/ReadVariableOpвCNN4_/Conv2D/ReadVariableOpвFC0_/BiasAdd/ReadVariableOpвFC0_/MatMul/ReadVariableOpвFC_2_/BiasAdd/ReadVariableOpвFC_2_/MatMul/ReadVariableOpИ
CNN0_/Conv2D/ReadVariableOpReadVariableOp$cnn0__conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ж
CNN0_/Conv2DConv2Dinputs#CNN0_/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
~
CNN0_/BiasAdd/ReadVariableOpReadVariableOp%cnn0__biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Р
CNN0_/BiasAddBiasAddCNN0_/Conv2D:output:0$CNN0_/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аг
MAX_POOL_0_/MaxPoolMaxPoolCNN0_/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
n
CNN_REL0_/ReluReluMAX_POOL_0_/MaxPool:output:0*
T0*/
_output_shapes
:         @И
CNN1_/Conv2D/ReadVariableOpReadVariableOp$cnn1__conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╗
CNN1_/Conv2DConv2DCNN_REL0_/Relu:activations:0#CNN1_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ *
paddingSAME*
strides
~
CNN1_/BiasAdd/ReadVariableOpReadVariableOp%cnn1__biasadd_readvariableop_resource*
_output_shapes
: *
dtype0П
CNN1_/BiasAddBiasAddCNN1_/Conv2D:output:0$CNN1_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ г
MAX_POOL_1_/MaxPoolMaxPoolCNN1_/BiasAdd:output:0*/
_output_shapes
:         @ *
ksize
*
paddingVALID*
strides
n
CNN_REL1_/ReluReluMAX_POOL_1_/MaxPool:output:0*
T0*/
_output_shapes
:         @ И
CNN2_/Conv2D/ReadVariableOpReadVariableOp$cnn2__conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0╗
CNN2_/Conv2DConv2DCNN_REL1_/Relu:activations:0#CNN2_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0*
paddingSAME*
strides
~
CNN2_/BiasAdd/ReadVariableOpReadVariableOp%cnn2__biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0П
CNN2_/BiasAddBiasAddCNN2_/Conv2D:output:0$CNN2_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @0г
MAX_POOL_2_/MaxPoolMaxPoolCNN2_/BiasAdd:output:0*/
_output_shapes
:          0*
ksize
*
paddingVALID*
strides
n
CNN_REL2_/ReluReluMAX_POOL_2_/MaxPool:output:0*
T0*/
_output_shapes
:          0И
CNN3_/Conv2D/ReadVariableOpReadVariableOp$cnn3__conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0╗
CNN3_/Conv2DConv2DCNN_REL2_/Relu:activations:0#CNN3_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @*
paddingSAME*
strides
~
CNN3_/BiasAdd/ReadVariableOpReadVariableOp%cnn3__biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0П
CNN3_/BiasAddBiasAddCNN3_/Conv2D:output:0$CNN3_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          @г
MAX_POOL_3_/MaxPoolMaxPoolCNN3_/BiasAdd:output:0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
n
CNN_REL3_/ReluReluMAX_POOL_3_/MaxPool:output:0*
T0*/
_output_shapes
:          @И
CNN4_/Conv2D/ReadVariableOpReadVariableOp$cnn4__conv2d_readvariableop_resource*&
_output_shapes
:@`*
dtype0╗
CNN4_/Conv2DConv2DCNN_REL3_/Relu:activations:0#CNN4_/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `*
paddingSAME*
strides
~
CNN4_/BiasAdd/ReadVariableOpReadVariableOp%cnn4__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0П
CNN4_/BiasAddBiasAddCNN4_/Conv2D:output:0$CNN4_/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          `г
MAX_POOL_4_/MaxPoolMaxPoolCNN4_/BiasAdd:output:0*/
_output_shapes
:         `*
ksize
*
paddingVALID*
strides
n
CNN_REL4_/ReluReluMAX_POOL_4_/MaxPool:output:0*
T0*/
_output_shapes
:         `м
AVG1_/AvgPoolAvgPoolCNN_REL4_/Relu:activations:0*
T0*/
_output_shapes
:         `*
ksize
*
paddingVALID*
strides
\
FLT1_/ConstConst*
_output_shapes
:*
dtype0*
valueB"    `   x
FLT1_/ReshapeReshapeAVG1_/AvgPool:output:0FLT1_/Const:output:0*
T0*'
_output_shapes
:         `~
FC0_/MatMul/ReadVariableOpReadVariableOp#fc0__matmul_readvariableop_resource*
_output_shapes

:``*
dtype0Г
FC0_/MatMulMatMulFLT1_/Reshape:output:0"FC0_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `|
FC0_/BiasAdd/ReadVariableOpReadVariableOp$fc0__biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Е
FC0_/BiasAddBiasAddFC0_/MatMul:product:0#FC0_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `_
FC_RELU0_/ReluReluFC0_/BiasAdd:output:0*
T0*'
_output_shapes
:         `А
FC_2_/MatMul/ReadVariableOpReadVariableOp$fc_2__matmul_readvariableop_resource*
_output_shapes

:`
*
dtype0Л
FC_2_/MatMulMatMulFC_RELU0_/Relu:activations:0#FC_2_/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
~
FC_2_/BiasAdd/ReadVariableOpReadVariableOp%fc_2__biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0И
FC_2_/BiasAddBiasAddFC_2_/MatMul:product:0$FC_2_/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
d
softmax/SoftmaxSoftmaxFC_2_/BiasAdd:output:0*
T0*'
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   
flatten/ReshapeReshapesoftmax/Softmax:softmax:0flatten/Const:output:0*
T0*'
_output_shapes
:         
g
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:         
я
NoOpNoOp^CNN0_/BiasAdd/ReadVariableOp^CNN0_/Conv2D/ReadVariableOp^CNN1_/BiasAdd/ReadVariableOp^CNN1_/Conv2D/ReadVariableOp^CNN2_/BiasAdd/ReadVariableOp^CNN2_/Conv2D/ReadVariableOp^CNN3_/BiasAdd/ReadVariableOp^CNN3_/Conv2D/ReadVariableOp^CNN4_/BiasAdd/ReadVariableOp^CNN4_/Conv2D/ReadVariableOp^FC0_/BiasAdd/ReadVariableOp^FC0_/MatMul/ReadVariableOp^FC_2_/BiasAdd/ReadVariableOp^FC_2_/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 2<
CNN0_/BiasAdd/ReadVariableOpCNN0_/BiasAdd/ReadVariableOp2:
CNN0_/Conv2D/ReadVariableOpCNN0_/Conv2D/ReadVariableOp2<
CNN1_/BiasAdd/ReadVariableOpCNN1_/BiasAdd/ReadVariableOp2:
CNN1_/Conv2D/ReadVariableOpCNN1_/Conv2D/ReadVariableOp2<
CNN2_/BiasAdd/ReadVariableOpCNN2_/BiasAdd/ReadVariableOp2:
CNN2_/Conv2D/ReadVariableOpCNN2_/Conv2D/ReadVariableOp2<
CNN3_/BiasAdd/ReadVariableOpCNN3_/BiasAdd/ReadVariableOp2:
CNN3_/Conv2D/ReadVariableOpCNN3_/Conv2D/ReadVariableOp2<
CNN4_/BiasAdd/ReadVariableOpCNN4_/BiasAdd/ReadVariableOp2:
CNN4_/Conv2D/ReadVariableOpCNN4_/Conv2D/ReadVariableOp2:
FC0_/BiasAdd/ReadVariableOpFC0_/BiasAdd/ReadVariableOp28
FC0_/MatMul/ReadVariableOpFC0_/MatMul/ReadVariableOp2<
FC_2_/BiasAdd/ReadVariableOpFC_2_/BiasAdd/ReadVariableOp2:
FC_2_/MatMul/ReadVariableOpFC_2_/MatMul/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
й

·
A__inference_CNN0__layer_call_and_return_conditional_losses_200680

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
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
:         Аh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ч
Ы
&__inference_CNN3__layer_call_fn_201626

inputs!
unknown:0@
	unknown_0:@
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_CNN3__layer_call_and_return_conditional_losses_200752w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          0
 
_user_specified_nameinputs
щ
a
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_201695

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         `:W S
/
_output_shapes
:         `
 
_user_specified_nameinputs
ў
В
&__inference_model_layer_call_fn_201341

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: 0
	unknown_4:0#
	unknown_5:0@
	unknown_6:@#
	unknown_7:@`
	unknown_8:`
	unknown_9:``

unknown_10:`

unknown_11:`


unknown_12:

identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_200965o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:         А: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
D
input_19
serving_default_input_1:0         А;
flatten0
StatefulPartitionedCall:0         
tensorflow/serving/predict:ги
┬
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer_with_weights-6
layer-20
layer-21
layer-22
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
6
 _init_input_shape"
_tf_keras_input_layer
▌
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op"
_tf_keras_layer
е
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
е
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op"
_tf_keras_layer
е
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
е
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
 S_jit_compiled_convolution_op"
_tf_keras_layer
е
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
е
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
 h_jit_compiled_convolution_op"
_tf_keras_layer
е
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
е
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

{kernel
|bias
 }_jit_compiled_convolution_op"
_tf_keras_layer
й
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer
л
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses
Ьkernel
	Эbias"
_tf_keras_layer
л
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layer
├
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses
кkernel
	лbias"
_tf_keras_layer
л
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"
_tf_keras_layer
л
▓	variables
│trainable_variables
┤regularization_losses
╡	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses"
_tf_keras_layer
К
'0
(1
<2
=3
Q4
R5
f6
g7
{8
|9
Ь10
Э11
к12
л13"
trackable_list_wrapper
К
'0
(1
<2
=3
Q4
R5
f6
g7
{8
|9
Ь10
Э11
к12
л13"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╦
╜trace_0
╛trace_1
┐trace_2
└trace_32╪
&__inference_model_layer_call_fn_200996
&__inference_model_layer_call_fn_201083
&__inference_model_layer_call_fn_201341
&__inference_model_layer_call_fn_201374╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0z╛trace_1z┐trace_2z└trace_3
╖
┴trace_0
┬trace_1
├trace_2
─trace_32─
A__inference_model_layer_call_and_return_conditional_losses_200854
A__inference_model_layer_call_and_return_conditional_losses_200908
A__inference_model_layer_call_and_return_conditional_losses_201437
A__inference_model_layer_call_and_return_conditional_losses_201500╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0z┬trace_1z├trace_2z─trace_3
╠B╔
!__inference__wrapped_model_200594input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
-
┼serving_default"
signature_map
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
т
╦trace_02├
&__inference_CNN0__layer_call_fn_201509Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
¤
╠trace_02▐
A__inference_CNN0__layer_call_and_return_conditional_losses_201519Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╠trace_0
&:$2CNN0_/kernel
:2
CNN0_/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ш
╥trace_02╔
,__inference_MAX_POOL_0__layer_call_fn_201524Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
Г
╙trace_02ф
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_201529Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╙trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ц
┘trace_02╟
*__inference_CNN_REL0__layer_call_fn_201534Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
Б
┌trace_02т
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_201539Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
т
рtrace_02├
&__inference_CNN1__layer_call_fn_201548Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zрtrace_0
¤
сtrace_02▐
A__inference_CNN1__layer_call_and_return_conditional_losses_201558Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0
&:$ 2CNN1_/kernel
: 2
CNN1_/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
ш
чtrace_02╔
,__inference_MAX_POOL_1__layer_call_fn_201563Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0
Г
шtrace_02ф
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_201568Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zшtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
ц
юtrace_02╟
*__inference_CNN_REL1__layer_call_fn_201573Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zюtrace_0
Б
яtrace_02т
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_201578Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
т
їtrace_02├
&__inference_CNN2__layer_call_fn_201587Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zїtrace_0
¤
Ўtrace_02▐
A__inference_CNN2__layer_call_and_return_conditional_losses_201597Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0
&:$ 02CNN2_/kernel
:02
CNN2_/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
ш
№trace_02╔
,__inference_MAX_POOL_2__layer_call_fn_201602Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z№trace_0
Г
¤trace_02ф
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_201607Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z¤trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
ц
Гtrace_02╟
*__inference_CNN_REL2__layer_call_fn_201612Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
Б
Дtrace_02т
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_201617Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zДtrace_0
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
т
Кtrace_02├
&__inference_CNN3__layer_call_fn_201626Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0
¤
Лtrace_02▐
A__inference_CNN3__layer_call_and_return_conditional_losses_201636Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЛtrace_0
&:$0@2CNN3_/kernel
:@2
CNN3_/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
ш
Сtrace_02╔
,__inference_MAX_POOL_3__layer_call_fn_201641Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0
Г
Тtrace_02ф
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_201646Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zТtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
ц
Шtrace_02╟
*__inference_CNN_REL3__layer_call_fn_201651Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
Б
Щtrace_02т
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_201656Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЩtrace_0
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
т
Яtrace_02├
&__inference_CNN4__layer_call_fn_201665Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЯtrace_0
¤
аtrace_02▐
A__inference_CNN4__layer_call_and_return_conditional_losses_201675Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zаtrace_0
&:$@`2CNN4_/kernel
:`2
CNN4_/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╢
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
~	variables
trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
ш
жtrace_02╔
,__inference_MAX_POOL_4__layer_call_fn_201680Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zжtrace_0
Г
зtrace_02ф
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_201685Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zзtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
ц
нtrace_02╟
*__inference_CNN_REL4__layer_call_fn_201690Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zнtrace_0
Б
оtrace_02т
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_201695Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zоtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
т
┤trace_02├
&__inference_AVG1__layer_call_fn_201700Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0
¤
╡trace_02▐
A__inference_AVG1__layer_call_and_return_conditional_losses_201705Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╡trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
т
╗trace_02├
&__inference_FLT1__layer_call_fn_201710Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0
¤
╝trace_02▐
A__inference_FLT1__layer_call_and_return_conditional_losses_201716Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╝trace_0
0
Ь0
Э1"
trackable_list_wrapper
0
Ь0
Э1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
с
┬trace_02┬
%__inference_FC0__layer_call_fn_201725Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0
№
├trace_02▌
@__inference_FC0__layer_call_and_return_conditional_losses_201735Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0
:``2FC0_/kernel
:`2	FC0_/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
ц
╔trace_02╟
*__inference_FC_RELU0__layer_call_fn_201740Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╔trace_0
Б
╩trace_02т
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_201745Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
0
к0
л1"
trackable_list_wrapper
0
к0
л1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
т
╨trace_02├
&__inference_FC_2__layer_call_fn_201754Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╨trace_0
¤
╤trace_02▐
A__inference_FC_2__layer_call_and_return_conditional_losses_201764Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
:`
2FC_2_/kernel
:
2
FC_2_/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
м	variables
нtrainable_variables
оregularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
ё
╫trace_02╥
(__inference_softmax_layer_call_fn_201769е
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╫trace_0
М
╪trace_02э
C__inference_softmax_layer_call_and_return_conditional_losses_201774е
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
▌layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
ф
▐trace_02┼
(__inference_flatten_layer_call_fn_201779Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▐trace_0
 
▀trace_02р
C__inference_flatten_layer_call_and_return_conditional_losses_201785Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0
 "
trackable_list_wrapper
╬
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
22"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
&__inference_model_layer_call_fn_200996input_1"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
&__inference_model_layer_call_fn_201083input_1"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
&__inference_model_layer_call_fn_201341inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
&__inference_model_layer_call_fn_201374inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
A__inference_model_layer_call_and_return_conditional_losses_200854input_1"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
A__inference_model_layer_call_and_return_conditional_losses_200908input_1"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
A__inference_model_layer_call_and_return_conditional_losses_201437inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
A__inference_model_layer_call_and_return_conditional_losses_201500inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╦B╚
$__inference_signature_wrapper_201308input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_CNN0__layer_call_fn_201509inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_CNN0__layer_call_and_return_conditional_losses_201519inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╓B╙
,__inference_MAX_POOL_0__layer_call_fn_201524inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_201529inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_CNN_REL0__layer_call_fn_201534inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_201539inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_CNN1__layer_call_fn_201548inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_CNN1__layer_call_and_return_conditional_losses_201558inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╓B╙
,__inference_MAX_POOL_1__layer_call_fn_201563inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_201568inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_CNN_REL1__layer_call_fn_201573inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_201578inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_CNN2__layer_call_fn_201587inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_CNN2__layer_call_and_return_conditional_losses_201597inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╓B╙
,__inference_MAX_POOL_2__layer_call_fn_201602inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_201607inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_CNN_REL2__layer_call_fn_201612inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_201617inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_CNN3__layer_call_fn_201626inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_CNN3__layer_call_and_return_conditional_losses_201636inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╓B╙
,__inference_MAX_POOL_3__layer_call_fn_201641inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_201646inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_CNN_REL3__layer_call_fn_201651inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_201656inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_CNN4__layer_call_fn_201665inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_CNN4__layer_call_and_return_conditional_losses_201675inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╓B╙
,__inference_MAX_POOL_4__layer_call_fn_201680inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_201685inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_CNN_REL4__layer_call_fn_201690inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_201695inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_AVG1__layer_call_fn_201700inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_AVG1__layer_call_and_return_conditional_losses_201705inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_FLT1__layer_call_fn_201710inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_FLT1__layer_call_and_return_conditional_losses_201716inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╧B╠
%__inference_FC0__layer_call_fn_201725inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
@__inference_FC0__layer_call_and_return_conditional_losses_201735inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_FC_RELU0__layer_call_fn_201740inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_201745inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╨B═
&__inference_FC_2__layer_call_fn_201754inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
A__inference_FC_2__layer_call_and_return_conditional_losses_201764inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
(__inference_softmax_layer_call_fn_201769inputs"е
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
C__inference_softmax_layer_call_and_return_conditional_losses_201774inputs"е
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╥B╧
(__inference_flatten_layer_call_fn_201779inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_flatten_layer_call_and_return_conditional_losses_201785inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ы
A__inference_AVG1__layer_call_and_return_conditional_losses_201705еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ┼
&__inference_AVG1__layer_call_fn_201700ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ║
A__inference_CNN0__layer_call_and_return_conditional_losses_201519u'(8в5
.в+
)К&
inputs         А
к "5в2
+К(
tensor_0         А
Ъ Ф
&__inference_CNN0__layer_call_fn_201509j'(8в5
.в+
)К&
inputs         А
к "*К'
unknown         А╕
A__inference_CNN1__layer_call_and_return_conditional_losses_201558s<=7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @ 
Ъ Т
&__inference_CNN1__layer_call_fn_201548h<=7в4
-в*
(К%
inputs         @
к ")К&
unknown         @ ╕
A__inference_CNN2__layer_call_and_return_conditional_losses_201597sQR7в4
-в*
(К%
inputs         @ 
к "4в1
*К'
tensor_0         @0
Ъ Т
&__inference_CNN2__layer_call_fn_201587hQR7в4
-в*
(К%
inputs         @ 
к ")К&
unknown         @0╕
A__inference_CNN3__layer_call_and_return_conditional_losses_201636sfg7в4
-в*
(К%
inputs          0
к "4в1
*К'
tensor_0          @
Ъ Т
&__inference_CNN3__layer_call_fn_201626hfg7в4
-в*
(К%
inputs          0
к ")К&
unknown          @╕
A__inference_CNN4__layer_call_and_return_conditional_losses_201675s{|7в4
-в*
(К%
inputs          @
к "4в1
*К'
tensor_0          `
Ъ Т
&__inference_CNN4__layer_call_fn_201665h{|7в4
-в*
(К%
inputs          @
к ")К&
unknown          `╕
E__inference_CNN_REL0__layer_call_and_return_conditional_losses_201539o7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ Т
*__inference_CNN_REL0__layer_call_fn_201534d7в4
-в*
(К%
inputs         @
к ")К&
unknown         @╕
E__inference_CNN_REL1__layer_call_and_return_conditional_losses_201578o7в4
-в*
(К%
inputs         @ 
к "4в1
*К'
tensor_0         @ 
Ъ Т
*__inference_CNN_REL1__layer_call_fn_201573d7в4
-в*
(К%
inputs         @ 
к ")К&
unknown         @ ╕
E__inference_CNN_REL2__layer_call_and_return_conditional_losses_201617o7в4
-в*
(К%
inputs          0
к "4в1
*К'
tensor_0          0
Ъ Т
*__inference_CNN_REL2__layer_call_fn_201612d7в4
-в*
(К%
inputs          0
к ")К&
unknown          0╕
E__inference_CNN_REL3__layer_call_and_return_conditional_losses_201656o7в4
-в*
(К%
inputs          @
к "4в1
*К'
tensor_0          @
Ъ Т
*__inference_CNN_REL3__layer_call_fn_201651d7в4
-в*
(К%
inputs          @
к ")К&
unknown          @╕
E__inference_CNN_REL4__layer_call_and_return_conditional_losses_201695o7в4
-в*
(К%
inputs         `
к "4в1
*К'
tensor_0         `
Ъ Т
*__inference_CNN_REL4__layer_call_fn_201690d7в4
-в*
(К%
inputs         `
к ")К&
unknown         `й
@__inference_FC0__layer_call_and_return_conditional_losses_201735eЬЭ/в,
%в"
 К
inputs         `
к ",в)
"К
tensor_0         `
Ъ Г
%__inference_FC0__layer_call_fn_201725ZЬЭ/в,
%в"
 К
inputs         `
к "!К
unknown         `к
A__inference_FC_2__layer_call_and_return_conditional_losses_201764eкл/в,
%в"
 К
inputs         `
к ",в)
"К
tensor_0         

Ъ Д
&__inference_FC_2__layer_call_fn_201754Zкл/в,
%в"
 К
inputs         `
к "!К
unknown         
и
E__inference_FC_RELU0__layer_call_and_return_conditional_losses_201745_/в,
%в"
 К
inputs         `
к ",в)
"К
tensor_0         `
Ъ В
*__inference_FC_RELU0__layer_call_fn_201740T/в,
%в"
 К
inputs         `
к "!К
unknown         `м
A__inference_FLT1__layer_call_and_return_conditional_losses_201716g7в4
-в*
(К%
inputs         `
к ",в)
"К
tensor_0         `
Ъ Ж
&__inference_FLT1__layer_call_fn_201710\7в4
-в*
(К%
inputs         `
к "!К
unknown         `ё
G__inference_MAX_POOL_0__layer_call_and_return_conditional_losses_201529еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╦
,__inference_MAX_POOL_0__layer_call_fn_201524ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ё
G__inference_MAX_POOL_1__layer_call_and_return_conditional_losses_201568еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╦
,__inference_MAX_POOL_1__layer_call_fn_201563ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ё
G__inference_MAX_POOL_2__layer_call_and_return_conditional_losses_201607еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╦
,__inference_MAX_POOL_2__layer_call_fn_201602ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ё
G__inference_MAX_POOL_3__layer_call_and_return_conditional_losses_201646еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╦
,__inference_MAX_POOL_3__layer_call_fn_201641ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ё
G__inference_MAX_POOL_4__layer_call_and_return_conditional_losses_201685еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╦
,__inference_MAX_POOL_4__layer_call_fn_201680ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    и
!__inference__wrapped_model_200594В'(<=QRfg{|ЬЭкл9в6
/в,
*К'
input_1         А
к "1к.
,
flatten!К
flatten         
ж
C__inference_flatten_layer_call_and_return_conditional_losses_201785_/в,
%в"
 К
inputs         

к ",в)
"К
tensor_0         

Ъ А
(__inference_flatten_layer_call_fn_201779T/в,
%в"
 К
inputs         

к "!К
unknown         
╦
A__inference_model_layer_call_and_return_conditional_losses_200854Е'(<=QRfg{|ЬЭклAв>
7в4
*К'
input_1         А
p

 
к ",в)
"К
tensor_0         

Ъ ╦
A__inference_model_layer_call_and_return_conditional_losses_200908Е'(<=QRfg{|ЬЭклAв>
7в4
*К'
input_1         А
p 

 
к ",в)
"К
tensor_0         

Ъ ╩
A__inference_model_layer_call_and_return_conditional_losses_201437Д'(<=QRfg{|ЬЭкл@в=
6в3
)К&
inputs         А
p

 
к ",в)
"К
tensor_0         

Ъ ╩
A__inference_model_layer_call_and_return_conditional_losses_201500Д'(<=QRfg{|ЬЭкл@в=
6в3
)К&
inputs         А
p 

 
к ",в)
"К
tensor_0         

Ъ д
&__inference_model_layer_call_fn_200996z'(<=QRfg{|ЬЭклAв>
7в4
*К'
input_1         А
p

 
к "!К
unknown         
д
&__inference_model_layer_call_fn_201083z'(<=QRfg{|ЬЭклAв>
7в4
*К'
input_1         А
p 

 
к "!К
unknown         
г
&__inference_model_layer_call_fn_201341y'(<=QRfg{|ЬЭкл@в=
6в3
)К&
inputs         А
p

 
к "!К
unknown         
г
&__inference_model_layer_call_fn_201374y'(<=QRfg{|ЬЭкл@в=
6в3
)К&
inputs         А
p 

 
к "!К
unknown         
╢
$__inference_signature_wrapper_201308Н'(<=QRfg{|ЬЭклDвA
в 
:к7
5
input_1*К'
input_1         А"1к.
,
flatten!К
flatten         
к
C__inference_softmax_layer_call_and_return_conditional_losses_201774c3в0
)в&
 К
inputs         


 
к ",в)
"К
tensor_0         

Ъ Д
(__inference_softmax_layer_call_fn_201769X3в0
)в&
 К
inputs         


 
к "!К
unknown         
