from collections import namedtuple, UserDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from functools import partial

from src.models.ecg_transformer import ECGTransformer
from src.utils.beamsearch import  BeamSearchScorer
from src.models.utils import get_1d_sincos_pos_embed
from src.utils.tensors import trunc_normal_

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

LARGE_NUMBER = 1e9


class ECGNMT(nn.Module):
    """"""
    def __init__(
        self, 
        vocab_size,
        vocab_idx,
        signal_size=2500,
        patch_size=10,
        in_chans=12,
        dmodel=768,
        depth=12,
        nhead=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        word_embedding=nn.Embedding,
        cls_token=False,        
        ):
        super().__init__()
        self.encoder = ECGTransformer(
            signal_size=signal_size,
            patch_size=patch_size,
            in_chans=in_chans,
            dmodel=dmodel,
            depth=depth,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            init_std=init_std,
            cls_token=cls_token
        )
        self.decoder = ECGDecoder(
            vocab_size=vocab_size,
            vocab_pad_idx=vocab_idx['<pad>'],
            dmodel=dmodel,
            depth=depth,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            word_embedding=word_embedding,
            init_std=init_std,
        )
        self.projection = nn.Linear(dmodel, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.vocab_pad_idx = vocab_idx['<pad>']
        self.vocab_eos_idx = vocab_idx['</s>']
        self.vocab_bos_idx = vocab_idx['<s>']
        
    def forward(self, x, tgt):
        """
        @param x(torch.FloatTensor of shape (B, C, L)) ecg input from then encoder
        @param tgt(torch.LongTensor of shape (B, N)) tgt tensor consisting of indices of words. ground truth values
        @param tgs(List[List[str]]) list of target sentence tokens, wrapped by `<s>` and `</s>`, and additionnal `<pad>` if needed:
                ground truth values 
        @return (torch.FloatTensor of shape (B, N, vocab_size)) logits on vocab size"""
        m = self.encoder(x)
        tgt_pad_masks = tgt == self.vocab_pad_idx
        x = self.decoder(tgt, m, tgt_pad_masks)
        x = self.projection(x)
        return x

    def step(self, m, tgt):
        """Call of one step of encoder + decoder. but do not perform the projection on vocab_size
        @param m (torch.FloatTensor of shape (b, n, d) output from the encoder
        @param tgt(torch.LongTensor of shape (b, n)) tgt tensor consisting of indices of words. ground truth values
        @return (torch.FloatTensor of shape (b, n, d)) output before projection on vocab_size"""
        tgt_pad_masks = tgt == self.vocab_pad_idx
        x = self.decoder(tgt, m, tgt_pad_masks, add_tgt_mask=False)
        return x


    def generate(self, x: torch.Tensor, num_beams: int = 5, max_decoding_time_step: int = 70):
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param x (torch.FloatTensor of shape (b, c, w) [b, 12, 2500]): batch of ecgs to generate
        @param num_beams (int): number of beam [or beam_size]
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses as returned by BeamSearchScorer.finalize
        """
        # src_ecgs = self.ecg_embedding_cnn(src_ecg).unsqueeze(0)
        
        if num_beams == 1:
            return self._greedy_search(x, max_decoding_time_step)
        else:
            return self._beam_search(x, num_beams, max_decoding_time_step)
        

    def _beam_search(self, x, num_beams, max_decoding_time_step):
        m = self.encoder(x)
        device = m.device
        b, g = m.size(0), 1
        
        beam_scorer = BeamSearchScorer(
            batch_size=b,
            num_beams=num_beams,
            device=device,
            max_length=max_decoding_time_step
        )

        tgt = torch.zeros((b*num_beams, 1), dtype=torch.long, device=device) + self.vocab_bos_idx # tensor version of [['<s>']]
        m = repeat(m, 'b enc_l d -> (b num_beams) enc_l d', num_beams=num_beams)
        
        # initialise score of first beam with 0 and the rest with -LARGE_NUMBER. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((b, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -LARGE_NUMBER
        beam_scores = rearrange(beam_scores, 'b num_beams -> (b num_beams)')
        t = 0
        while not beam_scorer.is_done and t < max_decoding_time_step:
            t += 1
            logits = self.projection(self.step(m, tgt)[:, -1, :])  # (b*num_beams, vocab_sz)
            log_p_t = F.log_softmax(logits, dim=-1)
            next_scores = log_p_t + repeat(beam_scores, 'b_nb -> b_nb vocab_sz', vocab_sz=self.vocab_size)
            next_scores = rearrange(next_scores, '(b num_beams) vocab_sz -> b (num_beams vocab_sz)', b=b)
            next_scores, next_tokens = next_scores.topk(k=2*num_beams, dim=-1)
            next_indices = torch.div(next_tokens, self.vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % self.vocab_size 

            out = beam_scorer.process(
                input_ids=tgt,
                next_scores=next_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
                pad_token_id=self.vocab_pad_idx,
                eos_token_id=self.vocab_eos_idx
            )

            beam_scores = out["next_beam_scores"]
            beam_next_tokens = out["next_beam_tokens"]
            beam_idx = out["next_beam_indices"]

            tgt = torch.cat([tgt[beam_idx, :], rearrange(beam_next_tokens, 'b_nb -> b_nb 1')], dim=-1)
        
        out = beam_scorer.finalize(
            input_ids=tgt,
            final_beam_scores=beam_scores,
            final_beam_tokens=next_tokens,
            final_beam_indices=next_indices,
            pad_token_id=self.vocab_pad_idx,
            eos_token_id=self.vocab_eos_idx,
            max_length=max_decoding_time_step+1,
        )         

        return out
    

    def _greedy_search(self, x, max_decoding_time_step):
        m = self.encoder(x)
        b = m.size(0)
        device = m.device
        unfinished_sequences = torch.ones(b, dtype=torch.long, device=device)
        scores = torch.zeros(b, device=device)
        tgt = torch.zeros((b, 1), dtype=torch.long, device=device) + self.vocab_bos_idx # tensor version of [['<s>']]
        
        t = 0
        while unfinished_sequences.max() > 0 and t < max_decoding_time_step:
            t += 1
            logits = self.projection(self.step(m, tgt)[:, -1, :])  # (b, vocab_sz)
            log_p_t = F.log_softmax(logits, dim=-1)
            next_scores, next_tokens = log_p_t.max(dim=-1)
            
            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + self.vocab_pad_idx * (1 - unfinished_sequences)
            scores +=  next_scores * unfinished_sequences
            tgt = torch.cat([tgt, rearrange(next_tokens, 'b -> b 1')], dim=-1)
            unfinished_sequences = unfinished_sequences & (next_tokens != self.vocab_eos_idx)
        
        return UserDict(
            {
                "sequences": tgt,
                "sequence_scores": scores,
            }
        )

    def generate_one(self, ecg: torch.Tensor, beam_size: int = 5, max_decoding_time_step: int = 70):
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param ecg (torch.FloatTensor of shape (c, w) [12, 2500]): a single ecg
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[int]: the decoded target sentence, represented as a list of words indexes
                score: float: the log-likelihood of the target sentence
        """
        # src_ecgs = self.ecg_embedding_cnn(src_ecg).unsqueeze(0)
        device = ecg.device
        ecgs = ecg.unsqueeze(0)
        m = self.encoder(ecgs)
        hypotheses = [[self.vocab_bos_idx]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            m = repeat(m, '1 n d -> hyp_num n d', hyp_num=hyp_num)
            tgt = torch.tensor([hyp[-1] for hyp in hypotheses], dtype=torch.long, device=device)
            y = self.step(m, tgt)

            # log probabilities over target words
            log_p_t = F.log_softmax(y, dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            continuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(continuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, self.vocab_size, rounding_mode='floor')
            hyp_word_ids = top_cand_hyp_pos % self.vocab_size

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                print('#'*20, prev_hyp_id, hyp_word_id, len(hypotheses))
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word_id]
                if hyp_word_id == self.vocab_eos_idx:
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    

class ECGDecoder(nn.Module):
    """"""
    def __init__(
        self, 
        vocab_size,
        vocab_pad_idx,
        dmodel,
        depth,
        nhead,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        word_embedding=nn.Embedding,
        init_std=0.02,
        **kwargs, 
        ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dmodel, 
            nhead=nhead,
            activation="gelu",
            dim_feedforward=int(mlp_ratio*dmodel),
            norm_first=True, 
            batch_first=True
            )
        max_len = 5000
        # self.pos_embed = PositionalEmbedding(dmodel)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dmodel), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], max_len,cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.word_embed = word_embedding(vocab_size, dmodel, vocab_pad_idx )
        self.decoder = nn.TransformerDecoder(decoder_layer, depth, norm=norm_layer(dmodel))
        
        # self.apply(lambda m: init_weights(m, init_std))
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

    def forward(self, tgt, m, tgt_pad_mask, add_tgt_mask=True):
        """
        @param tgt(torch.FloatTensor of shape (B, N, D)) N here is max generated length. ground truth values 
        @param x(torch.FloatTensor of shape (B, N, D)) input from encoder or memory from previous layer """
        
        tgt = self.word_embed(tgt)
        tgt = self.pos_embed(tgt)
    
        if add_tgt_mask: 
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), tgt.device)
            x = self.decoder(tgt, m, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        else:
            x = self.decoder(tgt, m, tgt_key_padding_mask=tgt_pad_mask)
        return x



def ecgnmt_tiny(patch_size=25, **kwargs):
    """Constructs a Vision Transformer for ECG signals.
    @param: patch_size (int): The size of each patch.
    @param: vocab_path: str
    @param: **kwargs: Additional arguments to be passed to the VisionTransformer constructor.
    @return:
        ECGTransformer: A 1D Vision Transformer model for processing ECG signals.
    """
    model = ECGNMT(
        patch_size=patch_size, dmodel=192, depth=12, nhead=3, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ecgnmt_model(model_name: str, vocab_size, vocab_idx, **kwargs):
    if model_name == 'ecgnmt_tiny': return ecgnmt_tiny(vocab_size=vocab_size, vocab_idx=vocab_idx, **kwargs)
    else: raise RuntimeError("model name should be ecgnmt_[large|base|small|tiny], not {}".format(model_name))
