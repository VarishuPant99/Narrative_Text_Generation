# Narrative_Text_Generation
NLG Task using LSTM, RNN, and GPT-2 models. Measuring accuracy using Grammarly.

About the Data:
Text file contains chapters from George R.R. Martin’s book series- 
Game of Thrones (A Song of Ice and Fire)
Size = 450 kb
Corpus Length=414584
Total Unique Words=9808
Total Unique Characters=40

Character level (lstm)

![image](https://user-images.githubusercontent.com/39706982/131303854-29a40703-7d6b-4915-94b9-c49332156a30.png)

Encode text and map each character to an integer and vice versa by creating 2 dictionaries-
![image](https://user-images.githubusercontent.com/39706982/131303874-6a906138-c631-46ba-b25f-6ab8e80dc287.png)
Input- One hot encoded characters , converted into a column vector
Function to One hot encode:
![image](https://user-images.githubusercontent.com/39706982/131303926-a3b46afe-db28-49de-bf41-345805e4477f.png)
Set 128 sequences per mini batch with 100 character steps per sequence
Architecture built using PyTorch
512 Hidden Neurons
Optimizer set as Adam
Learning Rate=0.001
Validation Fraction=0.1
Gradient Clipping =5
![image](https://user-images.githubusercontent.com/39706982/131303998-ea99a3f9-499a-4184-9fef-a75a3bb1c9e6.png)
Evaluation:
![image](https://user-images.githubusercontent.com/39706982/131304049-e29043ca-0ba1-4854-a52d-44cf11d1cfef.png)
Best Epoch =38
Minimum Validation Loss=1.5533
Perplexity =4.727
Sample 1: [Top K =3]
Lannister and the stanter the silver of the breathers of the rider of that wanted to he will batter than the cold with a side, but the moment sawed his feat as the mome tabe and the child of he stood the boys were said of all a side. The seaters was a smile. I want the bastard that was the back of his beat. Jon said a moment began to comprething.  The sing is not a man of the Winterfell to the king.  He said and shared his hands to the seven was all a seady of stangers and the streetthing walks

No. of Writing Issues =13
![image](https://user-images.githubusercontent.com/39706982/131304113-0e96e026-f401-4c54-9fc9-bb013f0be4b9.png)
![image](https://user-images.githubusercontent.com/39706982/131304122-97ccc351-d0a7-4b2d-9123-ac833248dd17.png)

CHARACTER LEVEL (RNN)
Vectorized Text
Sequence Length=100
Examples per epoch=4104
Batch Size=64
Learning Rate=0.001
Optimizer=Adam
Buffer Size=10000 (to shuffle the dataset) [TF data is designed to work with possibly infinite sequences, so it doesn't attempt to shuffle the entire sequence in memory. Instead , it maintains a buffer in which it shuffles elements]
Embedding Layers:The input layer. A trainable lookup table that will map the numbers of each character to a vector with 256 dimensions
1024 RNN Units
![image](https://user-images.githubusercontent.com/39706982/131304201-b0664592-e534-4ba4-9c71-d498795a50d0.png)
![image](https://user-images.githubusercontent.com/39706982/131304218-16d8f637-2bff-4944-98a8-5d4c3e478274.png)
![image](https://user-images.githubusercontent.com/39706982/131304223-6141df58-f555-45ff-bc1e-0587d76ec774.png)
Sample 2:Temperature 0.6
Lannister, but they did not scare him. He'd be the first I've had to friend.  He pulled at his clothes like an insistent loomed as it had cleaned all the blood off her hand, it turned away silently and snow, walking silent again.
So deep in thoughtfully provided. A princess, she thought, but she said,  Whatever you'd like to do, my prince. Jofftey reflection in the black pool, the septon was snoring softly, his head pillowed on an open book in front of him. Tyrion waited until the swinging had stoppe
![image](https://user-images.githubusercontent.com/39706982/131304278-582656b3-24e6-4b2c-a526-e655ee0024f1.png)
![image](https://user-images.githubusercontent.com/39706982/131304289-42b090c5-3b54-4e04-811e-fc3657da95f3.png)

Sentence generation using Word-Level lstm model

Split words based on white space.
Normalize all words to lowercase to reduce the vocabulary size.
Sequence Length = 30
Step Size = 3
Total Sequences = 25991
Optimizer=RMSprop
After vectorization, we train a 2 stacked LSTM architecture

![image](https://user-images.githubusercontent.com/39706982/131304336-e7a084f2-440d-4fb8-a04c-e85ecdcdf993.png)
![image](https://user-images.githubusercontent.com/39706982/131304351-1f1bb0fc-d6a5-407b-892e-67fb065b127d.png)
![image](https://user-images.githubusercontent.com/39706982/131304365-518ff6ec-b5f8-42f4-be18-849daa3a8448.png)
![image](https://user-images.githubusercontent.com/39706982/131304369-d89647e4-2a2f-41f2-a232-04c0bf30f365.png)
Trained for 50 epochs
Minimum Validation Loss=1.41
Perplexity=4.22

Seed:at the door of eddard stark. yet ned could not help but notice that those pleasures were taking a toll on the king. robert was breathing heavily by the time
Temperature=0.5
Sample:
 twenty reached way off on his hands for him by the house where that beyond up all, the lived out. ran until that bran just out all meet as front with nothing now, like asked as war little stark. but old door.
No. of writing issues:6
![image](https://user-images.githubusercontent.com/39706982/131304435-1225b4ab-9d1b-4443-9f0a-cb5da506de55.png)

GPT-2 Architecture
There are three released sizes of GPT-2[Generative Pretraining Transformer]-
124M (default): the "small" model, 500MB on disk. [WE USED THIS ONE]
355M: the "medium" model, 1.5GB on disk.
774M: the "large" model, cannot currently be finetuned with Collaboratory but can be used to generate text from the pretrained model (see later in Notebook)
1558M: the "extra large", true model. Will not work if a K80 GPU is attached to the notebook. (like 774M, it cannot be finetuned).
Larger models have more knowledge, but take longer to finetune and longer to generate text. 
Trained on 40 GB data scraped from the internet (called WebText)
Encoded text using Byte Pair Encoding 
Algorithm learns on it’s own and is not supervised for any particular task [loss works to predict next word] 
Still able to achieve State of the Art results in many language modelling tasks
Example- Translation, Summarization and Question Answering

Finetuning on our dataset


Learning rate for the training. (default 1e-4, we lowered to 1e-5 because we have <1MB input data)
Our dataset has 103968 tokens
Trained for 4000 steps, but loss stabilized after 3700 steps
Minimum Validation Loss=0.08
Minimum Perplexity=1.083
![image](https://user-images.githubusercontent.com/39706982/131304528-fd9a9ab2-43b1-4b23-8a22-441173d6e0e9.png)
![image](https://user-images.githubusercontent.com/39706982/131304538-bb4ef47f-30f6-4a16-90fd-706377afabd6.png)


Temperature=0.2
Lannister said, I don't think so. You're only a wolf, aren't you? Jon said, thinking how positively he looked at the feast day. He looked more like a giant in those early days, much like him. He was handsome and strong, yet somehow a little too handsome for his body, as if to give Jon his rightful place in the king's hall. Not anymore, Jon said. He wanted to be given back to the boy's world. Back that very moment he asked himself: Who has the right to his body? Be my brother's, Jon said. A question that will stay at the top of his lungs forever forever forever You were never born with my son, Royce said. I had a son with a pony at my table. He laughed. I had a daughter with a ponyboy. That wasn't my son, Jon said. An answer was in a word, but a third had not yet been given, and then more and more, and what was said were fierce words. They were Scarifs, said Jon, and he slammed the refocus back into place. 

Temperature=0.5
Lannister said, You're a bastard, aren't you? Oh, yes, Lannister said, astonished. Lannister was a hard boy to have muss thrown at him. I have proof, Lannister. I will bring Robert to you. Jon did not seem amused. I would bring Robert to you myself, he said. Lannister looked to Stark for guidance. He is my brother's room physician. DAENERYS
 Daenerys Targaryen looks forward to her time as Magister of Winterfell 
Arya: What is it, Jon? 
Jon: Catelyn. 
Catelyn: Are you sure that will change our plans for Robert? Jon looked at her. We might all be at the end of the world. Or, he said. Who knows, we might all be alive! Her eyes moved to Daenerys. Magistery is too important to me. You are my wife, she told him. I am your blood sister. And you must take me with you when you go off to war. 
![image](https://user-images.githubusercontent.com/39706982/131304598-141d4762-be6e-47ca-86bb-0b171525d343.png)

CONCLUSION AND Inferences

Accuracy of Word level model could be improved by stacking more LSTM models and increasing hidden layers
[Need more computational power]
This can be seen by the fact that Character Level Model using just a single LSTM does not fare well against Andrew Karpathy’s CharRNN which uses 1024 Gated Recurrent Neural Networks.
GPT-2 Architecture is very powerful and even finetuning with such a small dataset showed a great increase in accuracy.
[Can be experimented and tuned even more]
Accuracy metric for training – Perplexity
Accuracy metric for tuning sample generation- Grammarly [How legitimate?]


