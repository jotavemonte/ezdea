from scipy.optimize import linprog
import numpy


class DEA:
    def __init__(self, matriz, n_inputs, n_outputs):
        self.matriz = matriz
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_dmus = len(self.matriz)

    def _tornar_inputs_negativos(self):
        matriz_ajustada = []
        for linha in self.matriz:
            matriz_ajustada.append([-x for x in linha[0:self.n_inputs]]
                                   +
                                   linha[self.n_inputs::])
        return matriz_ajustada

    def _resposta_limpa_primal(self, res_dea):
        resp_dic = {}
        resp_dic['.func_obj'] = res_dea['fun']
        base_len = self.n_inputs + self.n_outputs
        for i in range(self.n_inputs):
            resp_str = f'u{i + 1}'
            resp_dic[resp_str] = res_dea['x'][i]
        for i in range(self.n_inputs, self.n_inputs + self.n_outputs):
            resp_str = f'v{i - self.n_inputs + 1}'
            resp_dic[resp_str] = res_dea['x'][i]
        if len(res_dea['x']) > base_len:
            resp_dic["f'1"] = res_dea['x'][base_len]
            resp_dic["f'2"] = res_dea['x'][base_len + 1]
            resp_dic['F'] = resp_dic["f'1"] - resp_dic["f'2"]
        return resp_dic

    def _resposta_limpa_dual(self, res_dea):
        resp_dic = {}
        resp_dic['.func_obj'] = res_dea['fun']
        for i in range(len(res_dea['x'])):
            if i == 0:
                resp_str = 'θ'
            else:
                if i < 10:
                    resp_str = f'λ0{i}'
                else:
                    resp_str = f'λ{i}'
            resp_dic[resp_str] = res_dea['x'][i]
            # resp_dic['slack_'+resp_str] = res_dea['slack'][i]
        return resp_dic

    def _resposta_mtx(self, res_dea):
        aux = res_dea['x']
        fun = numpy.array([res_dea['fun']])
        vec = numpy.concatenate((fun, aux), axis=0)
        return list(vec)

    @staticmethod
    def _valida_mtx(entrada):
        if not isinstance(entrada[0], type(list())):
            raise TypeError('Matrix Expected')

    def ccr_primal_input(self, matrix=False):
        ccr_aub = self._tornar_inputs_negativos()
        solucao = []
        for linha in self.matriz:
            # max_ = [inputs = 0, outputs = -outputs]
            max_ = [0 for x in linha[0:self.n_inputs]]
            max_ += [-x for x in linha[self.n_inputs::]]
            # linear
            linear = [x for x in linha[0:self.n_inputs]]
            linear += [0 for x in linha[self.n_inputs::]]
            res_dea = linprog(
                max_,
                ccr_aub,
                [0 for x in range(self.n_dmus)],
                [linear],
                [1]
                )
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear"
                    )
            res_dea['fun'] = - res_dea['fun']
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_primal(res_dea))
        return solucao

    def ccr_primal_output(self, matrix=False):
        ccr_aub = self._tornar_inputs_negativos()
        solucao = []
        for linha in self.matriz:
            min_ = linha[0:self.n_inputs] + [0 for x in linha[self.n_inputs:]]
            linear = [0 for x in linha[0:self.n_inputs]]
            linear += linha[self.n_inputs:]
            res_dea = linprog(
                min_,
                ccr_aub,
                [0 for x in range(self.n_dmus)],
                [linear],
                [1])
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear"
                    )
            res_dea['fun'] = 1/res_dea['fun']
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_primal(res_dea))
        return solucao

    def ccr_dual_input(self, matrix=False):
        matriz_aub_t = numpy.transpose(self.matriz)
        rest_y = []
        for index in range(self.n_inputs,
                           self.n_inputs + self.n_outputs):
            aux = [0] + [-x for x in matriz_aub_t[index]]
            rest_y.append(aux)
        solucao = []
        for linha in self.matriz:
            min_ = [1] + [0 for x in range(self.n_dmus)]
            rest_x = []
            for index in range(self.n_inputs):
                aux = [-linha[index]] + [x for x in matriz_aub_t[index]]
                rest_x.append(aux)
            rest = rest_x + rest_y
            vet_ineq = [0 for x in range(self.n_inputs)]
            vet_ineq += [-x for x in linha[self.n_inputs:]]
            res_dea = linprog(min_, rest, vet_ineq)
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear")
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_dual(res_dea))
        return solucao

    def ccr_dual_output(self, matrix=False):
        matriz_aub_t = numpy.transpose(self.matriz)
        solucao = []
        rest_x = []
        for index in range(self.n_inputs):
            aux = [0] + [x for x in matriz_aub_t[index]]
            rest_x.append(aux)
        for linha in self.matriz:
            max_ = [-1] + [0 for x in range(0, self.n_dmus)]
            rest_y = []
            for index in range(self.n_inputs, self.n_inputs + self.n_outputs):
                aux = [linha[index]] + [-x for x in matriz_aub_t[index]]
                rest_y.append(aux)
            rest = rest_x + rest_y
            vet_ineq = [x for x in linha[0:self.n_inputs]]
            vet_ineq += [
                0 for x in range(
                    self.n_inputs, self.n_inputs + self.n_outputs
                    )
                ]
            res_dea = linprog(max_, rest, vet_ineq)
            res_dea['x'][0] = 1/res_dea['x'][0]
            res_dea['fun'] = -1/res_dea['fun']
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear")
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_dual(res_dea))
        return solucao

    def bcc_primal_input(self, matrix=False):
        bcc_aub = self._tornar_inputs_negativos()
        bcc_aub = [(x + [1, -1]) for x in bcc_aub]
        solucao = []
        for linha in self.matriz:
            max_ = [0 for x in linha[0:self.n_inputs]]
            max_ += [-x for x in linha[self.n_inputs:]] + [-1, 1]
            linear = linha[0:self.n_inputs]
            linear += [0 for x in linha[self.n_inputs:]] + [0, 0]
            res_dea = linprog(
                max_,
                bcc_aub,
                [0 for x in range(self.n_dmus)],
                [linear],
                [1]
                )
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear"
                    )
            res_dea['fun'] = -res_dea['fun']
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_primal(res_dea))
        return solucao

    def bcc_primal_output(self, matrix=False):
        bcc_aub = self._tornar_inputs_negativos()
        bcc_aub = [(x + [-1, 1]) for x in bcc_aub]
        solucao = []
        for linha in self.matriz:
            min_ = linha[0:self.n_inputs]
            min_ += [0 for x in linha[self.n_inputs:]] + [1, -1]
            linear = [0 for x in linha[0:self.n_inputs]]
            linear += linha[self.n_inputs:] + [0, 0]
            res_dea = linprog(
                min_,
                bcc_aub,
                [0 for x in range(self.n_dmus)],
                [linear],
                [1])
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear"
                    )
            res_dea['fun'] = 1/res_dea['fun']
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_primal(res_dea))
        return solucao

    def bcc_dual_input(self, matrix=False):
        func_matriz = [(x + [1, -1]) for x in self.matriz]
        matriz_aub_t = numpy.transpose(func_matriz)
        rest_y = []
        for index in range(self.n_inputs,
                           self.n_inputs + self.n_outputs):
            aux = [0] + [-x for x in matriz_aub_t[index]]
            rest_y.append(aux)
        solucao = []
        for linha in func_matriz:
            min_ = [1] + [0 for x in range(self.n_dmus)]
            rest_x = []
            for index in range(self.n_inputs):
                aux = [-linha[index]] + [x for x in matriz_aub_t[index]]
                rest_x.append(aux)
            rest = rest_x + rest_y
            vet_ineq = [0 for x in range(self.n_inputs)]
            vet_ineq += [-x for x in linha[
                self.n_inputs:self.n_inputs + self.n_outputs
            ]]
            linear = [0] + [1 for x in matriz_aub_t[0]]
            res_dea = linprog(min_, rest, vet_ineq, [linear], [1])
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear")
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_dual(res_dea))
        return solucao

    def bcc_dual_output(self, matrix=False):
        func_matriz = [(x + [1, -1]) for x in self.matriz]
        matriz_aub_t = numpy.transpose(func_matriz)
        solucao = []
        rest_x = []
        for index in range(self.n_inputs):
            aux = [0] + [x for x in matriz_aub_t[index]]
            rest_x.append(aux)
        for linha in func_matriz:
            max_ = [-1] + [0 for x in range(0, self.n_dmus)]
            rest_y = []
            for index in range(self.n_inputs, self.n_inputs + self.n_outputs):
                aux = [linha[index]] + [-x for x in matriz_aub_t[index]]
                rest_y.append(aux)
            rest = rest_x + rest_y
            vet_ineq = [x for x in linha[0:self.n_inputs]]
            vet_ineq += [
                0 for x in range(
                    self.n_inputs, self.n_inputs + self.n_outputs
                    )
                ]
            linear = [0] + [1 for x in matriz_aub_t[0]]
            res_dea = linprog(max_, rest, vet_ineq, [linear], [1])
            res_dea['x'][0] = 1/res_dea['x'][0]
            res_dea['fun'] = -1/res_dea['fun']
            if res_dea['success'] is False:
                raise Exception(
                    'InputError',
                    "The input matrix don't returns a valid answear")
            if matrix:
                solucao.append(self._resposta_mtx(res_dea))
            else:
                solucao.append(self._resposta_limpa_dual(res_dea))
        return solucao

    @staticmethod
    def _soma_produto(linha_a, linha_b):
        valor_total = 0
        for x, y in zip(linha_a, linha_b):
            valor_total += x*y
        return valor_total

    def calcular_metas(self, dual_output):
        dual_formatada = [x[2:] for x in dual_output]
        matriz_transposta = numpy.array(self.matriz).transpose()
        metas = []
        for linha_matriz in matriz_transposta:
            linha_metas = []
            for linha_dual in dual_formatada:
                soma_produto = self._soma_produto(linha_matriz, linha_dual)
                linha_metas.append(soma_produto)
            metas.append(linha_metas)
        return metas

