import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        
        Code heavily copied from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    # top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k

        current_logits = torch.squeeze(logits[-1],0)
        indices_to_remove = current_logits < torch.topk(current_logits, top_k)[0][..., -1, None]
        current_logits[indices_to_remove] = filter_value
        logits[-1][0] = current_logits

    if top_p > 0.0:
        current_logits = torch.squeeze(logits[-1],0)
        sorted_logits, sorted_indices = torch.sort(current_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        current_logits[indices_to_remove] = filter_value
        logits[-1][0] = current_logits

    return logits

def top_k_top_p_sampling(x, encoder, decoder,
                embeddings, classifier):
    """ Performing topk or topp (Nucleus Sampling) as it is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Pure Temperture Sampling is left as comment for now.
        Not implemented: (1) Pass topp or topk as argument. 
                         
    """


    bs, _, _ = x.size()
    device = x.device

    decoded_batch = []
    sos_token = 0
    eos_token = 9
    max_length = 22

    temperature = 0.7
    top_k = 0
    top_p = 0.9


    for idx in range(bs):

        encoder_outputs = encoder(x).permute(1, 0, 2)
        encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)  # (time_step, 1, out_channels)
        decoder_input = torch.LongTensor([[sos_token]]).to(device)
        decoded_tokens = [decoder_input]
        caption_length = 1

        # Stop decoding after eos is decoded or when the desired max caption lenght is reached
        while decoder_input[-1].item() != eos_token and caption_length < max_length:
            embedding = embeddings(decoder_input)  # (num_words,1, out_channel)
            decoder_output = decoder(
                    embedding, encoder_output, attention_mask=None)
            logits = classifier(decoder_output)

            # Divide current output logits with temperture if > 1
            if temperature < 1:
                logits[-1][0] = logits[-1][0] / temperature

            # filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            # Pure Temperature Sampling
            filtered_logits = logits
            # print(filtered_logits.shape)

            torch.set_printoptions(edgeitems=100000)

            probabilities = F.softmax(filtered_logits, dim=-1)

            # Sample next token from the filterd destribution
            next_token = torch.multinomial(probabilities[-1], 1)
            decoded_tokens.append(next_token)
            decoder_input = torch.squeeze(torch.stack(decoded_tokens), 1)
            caption_length += 1

        decoded_batch.append(torch.stack(decoded_tokens))
    return pad_sequence(decoded_batch, batch_first=False, padding_value=eos_token)[:, :, -1]