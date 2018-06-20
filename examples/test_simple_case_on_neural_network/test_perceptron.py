#!/usr/bin/env python

import numpy

import Perceptron
import ForwardPropagator

#	globals
Tables_AND = ( #	inputs
				numpy.matrix( [ [ 0, 0 ], 
								[ 0, 1 ],
								[ 1, 0 ],
								[ 1, 1 ] ] ),
				#	outputs
				numpy.matrix( [ [ 0 ], 
								[ 0 ],
								[ 0 ],
								[ 1 ] ] ) )


InputsMatrix = Tables_AND[0]
OuputsMatrix = Tables_AND[1]
NumHiddenLayersNeurons = 2

NumInputs = InputsMatrix.shape[1]
NumOuputs = OuputsMatrix.shape[1]

#	main
def main():
	print( 'main()...........' )
	print( '           NumInputs = {}'.format( NumInputs ) )
	print( '           NumHiddenLayersNeurons = {}'.format( NumHiddenLayersNeurons ) )
	print( '           NumOuputs = {}'.format( NumOuputs ) )
	
	print( '           InputsMatrix = {}'.format( InputsMatrix ) )
	print( '           OuputsMatrix = {}'.format( OuputsMatrix ) )
	
	#	create perceptron
	perceptron = Perceptron.Perceptron( NumInputs, NumHiddenLayersNeurons, NumOuputs )
	print( 'PERCEPTRON' )
	print( '   hidden layer neurons' )
	print( '            perceptron.hiddenLayerWeightsMatrix = {}'.format( perceptron.hiddenLayerWeightsMatrix ) )
	print( '            perceptron.hiddenLayerBiasMatrix = {}'.format( perceptron.hiddenLayerBiasMatrix ) )
	print( '   outputs' )
	print( '            perceptron.outputLayerWeightsMatrix = {}'.format( perceptron.outputLayerWeightsMatrix ) )
	print( '            perceptron.outputLayerBiasMatrix = {}'.format( perceptron.outputLayerBiasMatrix ) )
	
	#	create forward propagate
	forwardPropagate = ForwardPropagator.ForwardPropagator( InputsMatrix )
	
	#	do forward propagate
	forwardPropagate.propagate( perceptron )
	print( 'FORWARD PROPAGATE' )
	print( '            forwardPropagate.inputsMatrix = {}'.format( forwardPropagate.inputsMatrix ) )
	print( '            forwardPropagate.outputsMatrixHiddenLayerNeurons = {}'.format( forwardPropagate.outputsMatrixHiddenLayerNeurons ) )
	print( '            forwardPropagate.outputsMatrixOutputsLayer = {}'.format( forwardPropagate.outputsMatrixOutputsLayer ) )
	
	
	
	
if __name__ == '__main__':
	main()
