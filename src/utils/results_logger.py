import logging
import os
import pickle
import csv
import numpy as np

class Logger:
    def __init__(self, logdir, rank, type='torch', debug=False, filename=None, summary=True, step=None, name=None):
        self.writer = None
        self.type = type
        self.rank = rank
        self.step = step
        self.logdir = logdir
        self.logdir_results = os.path.join(logdir, "cumulative")
        self.summary = summary and rank == 0
        if summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.logdir_results)
        else:
            self.type = 'None'

        self.debug_flag = debug
        logging.basicConfig(filename=filename, level=logging.INFO, format=f'%(levelname)s:{rank}: %(message)s', force=True)

        if rank == 0:
            logging.info(f"[!] starting logging at directory {logdir}")
            if self.debug_flag:
                logging.info(f"[!] Entering DEBUG mode")

    def commit(self, intermediate=False):
        pass

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.info("Closing the Logger.")

    def add_scalar(self, tag, scalar_value, step=None, intermediate=False):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.writer.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None, intermediate=False):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.writer.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None, intermediate=False):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            self.writer.add_figure(tag, image, step)

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag

    def is_not_none(self):
        return self.type != "None"

    def add_config(self, opts):
        self.add_table("Opts", vars(opts))

    def add_table(self, tag, tbl, step=None):
        if self.is_not_none():
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.writer.add_text(tag, tbl_str, step)

    def print(self, msg):
        logging.info(msg)

    def info(self, msg):
        if self.rank == 0:
            logging.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            logging.info(msg)

    def error(self, msg):
        logging.error(msg)

    def save_cumulative_results(self):
        # rec prec ap
        text = ''
        with open(self.logdir + '/rec_prec_ap.csv') as f:
            data = list(csv.reader(f, delimiter=','))
            for idx, row in enumerate(data):
                text += f"<tr><td>{row[0]}</td>" + f"<td>{row[1]}</td>" + f"<td>{row[2]}</td>" + f"<td>{row[3]}</td></tr>"
                if idx == len(data)-1:
                    urec = float(row[1])
                    upre = float(row[2])
                    uf1 = (2 * urec * upre)/(urec+upre)
            text += "</table>"
        self.writer.add_text('REC_PREC_AP', text, 0)

        # udr udp
        with open(self.logdir + '/udr_udp.csv') as f:
            data = csv.reader(f, delimiter=',')
            rows = [row for row in data]
            udr = float(rows[1][0])
            udp = float(rows[1][1])
            udf1 = (2*udr*udp)/(udr+udp)
        # wi
        with open(self.logdir+'/wi.pkl', 'rb') as f:
            data = pickle.load(f)
            wi = np.asarray(list(data.values())).mean()

        # wi no simplified
        with open(self.logdir+'/wi_no_simplified.pkl', 'rb') as f:
            data = pickle.load(f)
            wi_no_simplified = np.asarray(list(data.values())).mean()

        # wi_adjusted
        with open(self.logdir+'/wi_adjusted.pkl', 'rb') as f:
            data = pickle.load(f)
            wi_adjusted = np.asarray(list(data.values())).mean()

        # a_ose
        with open(self.logdir + '/a_ose.pkl', 'rb') as f:
            data = pickle.load(f)
            a_ose = 0.
            for v in data[:-1]:
                a_ose += v[0]

        # cumulative
        text = ''+f"<tr><td>U-RECALL</td><td>U-PRECISION</td><td>U-F1</td><td>UD-R</td>" \
                  f"<td>UD-P</td><td>UD-F1</td><td>WI</td><td>WI_NO_SIMPLIFIED</td><td>WI-ADJUSTED</td><td>A-OSE</td></tr>"
        text += f"<tr><td>{urec*100}</td>" + f"<td>{upre*100}</td>" + f"<td>{uf1*100}</td>" + f"<td>{udr*100}</td>" + \
                f"<td>{udp*100}</td>" + f"<td>{udf1*100}</td>" + f"<td>{wi*100}</td>" + f"<td>{wi_no_simplified*100}</td>" + f"<td>{wi_adjusted*100}</td>" + \
                f"<td>{a_ose}</td></tr>"
        text += "</table>"
        self.writer.add_text('CUMULATIVE', text, 0)

