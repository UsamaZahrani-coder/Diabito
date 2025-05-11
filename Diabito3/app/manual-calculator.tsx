import React, { useState } from 'react';
import { StyleSheet, View, Text, TextInput, Button, ActivityIndicator, Alert, ScrollView, SafeAreaView, TouchableOpacity, Image } from 'react-native';
import { router } from 'expo-router';

interface CalculationResult {
  correction_dose: number;
  carb_dose: number;
  bolus_dose: number;
  bolus_dose_ceil: number;
}

export default function ManualCalculator() {
  const [glucose, setGlucose] = useState('');
  const [targetGlucose, setTargetGlucose] = useState('');
  const [insulinSensitivity, setInsulinSensitivity] = useState('');
  const [carbInsulinRatio, setCarbInsulinRatio] = useState('');
  const [carbs, setCarbs] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CalculationResult | null>(null);

  const calculateInsulin = () => {
    if (!glucose || !targetGlucose || !insulinSensitivity || !carbInsulinRatio || !carbs) {
      Alert.alert('Error', 'Please fill all fields');
      return;
    }
    const API_URL = "http://192.168.0.111:5000";


    setLoading(true);

    try {
      // Convert inputs to numbers
      const glucoseValue = parseFloat(glucose);
      const targetValue = parseFloat(targetGlucose);
      const sensitivityValue = parseFloat(insulinSensitivity);
      const ratioValue = parseFloat(carbInsulinRatio);
      const carbsValue = parseFloat(carbs);

      // Calculate correction dose
      const correctionDose = (glucoseValue - targetValue) / sensitivityValue;

      // Calculate carb dose
      const carbDose = carbsValue / ratioValue;

      // Calculate total bolus dose
      const bolusDose = correctionDose + carbDose;

      // Round up to the nearest whole number
      const bolusDoseCeil = Math.ceil(bolusDose);

      setResult({
        correction_dose: parseFloat(correctionDose.toFixed(2)),
        carb_dose: parseFloat(carbDose.toFixed(2)),
        bolus_dose: parseFloat(bolusDose.toFixed(2)),
        bolus_dose_ceil: bolusDoseCeil
      });
    } catch (error) {
      Alert.alert('Error', 'Invalid input values. Please check your entries.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={{ flex: 1 }}>
        <View style={styles.content}>
          <TouchableOpacity 
            style={styles.backButton}
            onPress={() => router.back()}
          >
            <Text style={styles.backButtonText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Manual Insulin Calculator</Text>
          <View style={styles.form}>
            <TextInput
              style={styles.input}
              placeholder="Current Glucose Level"
              value={glucose}
              onChangeText={setGlucose}
              keyboardType="numeric"
            />
            <TextInput
              style={styles.input}
              placeholder="Target Glucose Level"
              value={targetGlucose}
              onChangeText={setTargetGlucose}
              keyboardType="numeric"
            />
            <TextInput
              style={styles.input}
              placeholder="Insulin Sensitivity Factor"
              value={insulinSensitivity}
              onChangeText={setInsulinSensitivity}
              keyboardType="numeric"
            />
            <TextInput
              style={styles.input}
              placeholder="Carb to Insulin Ratio"
              value={carbInsulinRatio}
              onChangeText={setCarbInsulinRatio}
              keyboardType="numeric"
            />
            <TextInput
              style={styles.input}
              placeholder="Carbohydrates (grams)"
              value={carbs}
              onChangeText={setCarbs}
              keyboardType="numeric"
            />
            
            <View style={styles.calculateButtonContainer}>
              <TouchableOpacity style={styles.calculateButton} onPress={calculateInsulin}>
                <Text style={styles.calculateButtonText}>Calculate Insulin</Text>
              </TouchableOpacity>
            </View>
            
            {loading && <ActivityIndicator size="large" color="#007AFF" style={styles.loader} />}
            
            {result && (
              <View style={styles.resultContainer}>
                <Text style={[styles.resultTitle, styles.secondTitle]}>Insulin Doses</Text>
                <View style={styles.resultSection}>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Correction Dose:</Text>
                    <Text style={styles.resultValue}>{result.correction_dose} units</Text>
                  </View>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Carb Dose:</Text>
                    <Text style={styles.resultValue}>{result.carb_dose} units</Text>
                  </View>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Total Bolus Dose:</Text>
                    <Text style={styles.resultValue}>{result.bolus_dose} units</Text>
                  </View>
                  <View style={[styles.resultRow, styles.finalResult]}>
                    <Text style={[styles.resultLabel, styles.boldText]}>Rounded Bolus Dose:</Text>
                    <Text style={[styles.resultValue, styles.boldText]}>{result.bolus_dose_ceil} units</Text>
                  </View>
                </View>
              </View>
            )}
          </View>
        </View>
      </ScrollView>
      <View style={styles.bottomNav}>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/')}
        >
          <Image 
            source={require('../icons/home.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/calculator')}
        >
          <Image 
            source={require('../icons/calculator_clicked.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/chat')}
        >
          <Image 
            source={require('../icons/chat.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/profile')}
        >
          <Image 
            source={require('../icons/Profile.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  bottomNav: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: '#fff',
    height: 60,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: '#e1e1e1'
  },
  navItem: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center'
  },
  navIcon: {
    width: 24,
    height: 24,
    resizeMode: 'contain'
  },
  container: {
    flex: 1,
    backgroundColor: '#f5f6fa'
  },
  content: {
    padding: 20
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 20,
    textAlign: 'center'
  },
  form: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 15,
    marginBottom: 15,
    fontSize: 16
  },
  calculateButtonContainer: {
    marginVertical: 20
  },
  calculateButton: {
    backgroundColor: '#000',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center'
  },
  calculateButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600'
  },
  loader: {
    marginVertical: 20
  },
  resultContainer: {
    marginTop: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 15
  },
  secondTitle: {
    marginTop: 20
  },
  resultSection: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 12
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee'
  },
  resultLabel: {
    fontSize: 14,
    color: '#2c3e50'
  },
  resultValue: {
    fontSize: 14,
    color: '#2c3e50',
    fontWeight: '500'
  },
  finalResult: {
    marginTop: 8,
    borderBottomWidth: 0,
    backgroundColor: '#e3f2fd',
    borderRadius: 6,
    padding: 8
  },
  boldText: {
    fontWeight: 'bold',
    color: '#1976d2'
  },
  backButton: {
    marginBottom: 20,
    alignSelf: 'flex-start'
  },
  backButtonText: {
    fontSize: 24,
    color: '#000',
    fontWeight: '500'
  }
});