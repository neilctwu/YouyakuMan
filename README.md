# YouyakuMan 

 [![Unstable](https://poser.pugx.org/ali-irawan/xtra/v/unstable.svg)](*https://poser.pugx.org/ali-irawan/xtra/v/unstable.svg*)  [![License](https://poser.pugx.org/ali-irawan/xtra/license.svg)](*https://poser.pugx.org/ali-irawan/xtra/license.svg*) 

### Introduction

This is an one-touch extractive summarization machine.

using BertSum as summatization model, extract top N important sentences.

### Example

```
$python youyakuman.py -txt_file YOUR_FILE -n 3
```

### Note

Since Bert only takes 512 length as inputs, this summarizer crop articles >512 length.

If --super_long option is used, summarizer automatically parse to numbers of 512 length inputs and summarize per inputs. Number of extraction might slightly altered with --super_long used.

