import React, { useState } from 'react';
import { StyleSheet, View, Text, TextInput, TouchableOpacity, Alert, Image } from 'react-native';
import { router } from 'expo-router';
import axios from 'axios';

const API_URL = "http://192.168.0.111:5000";

export default function Signup() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [errors, setErrors] = useState({
    password: '',
    confirmPassword: '',
    general: ''
  });

  const validatePassword = (password: string) => {
    if (password.length < 12) {
      return 'Password must be at least 12 characters long';
    }
    if (!/\d/.test(password)) {
      return 'Password must contain at least one number';
    }
    return null;
  };

  const handleSignup = async () => {
    setErrors({ password: '', confirmPassword: '', general: '' });
    
    // Trim whitespace from input fields
    const trimmedUsername = username.trim();
    const trimmedEmail = email.trim();
    const trimmedPassword = password.trim();
    const trimmedConfirmPassword = confirmPassword.trim();

    // Update state with trimmed values
    setUsername(trimmedUsername);
    setEmail(trimmedEmail);
    setPassword(trimmedPassword);
    setConfirmPassword(trimmedConfirmPassword);
    
    const passwordError = validatePassword(trimmedPassword);
    if (passwordError) {
      setErrors(prev => ({ ...prev, password: passwordError }));
      return;
    }

    if (trimmedPassword !== trimmedConfirmPassword) {
      setErrors(prev => ({ ...prev, confirmPassword: 'Passwords do not match' }));
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/register`, {
        username: trimmedUsername,
        email: trimmedEmail,
        password: trimmedPassword
      });

      Alert.alert('Success', 'Account created successfully', [
        { text: 'OK', onPress: () => router.replace('/login') }
      ]);
    } catch (error: any) {
      setErrors(prev => ({ ...prev, general: error.response?.data?.error || 'Registration failed' }));
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Create Account</Text>
      <TextInput
        style={styles.input}
        placeholder="Username"
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="Email"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        keyboardType="email-address"
      />
      <View style={styles.passwordContainer}>
        <TextInput
          style={[styles.input, styles.passwordInput, errors.password ? styles.inputError : null]}
          placeholder="Password"
          value={password}
          onChangeText={(text) => {
            setPassword(text);
            setErrors(prev => ({ ...prev, password: '' }));
          }}
          secureTextEntry={!showPassword}
        />
        <TouchableOpacity
          style={styles.eyeButton}
          onPress={() => setShowPassword(!showPassword)}
        >
          <Image
            source={showPassword ? require('../icons/seePassword.png') : require('../icons/toggleoffpassword.png')}
            style={styles.eyeIcon}
          />
        </TouchableOpacity>
      </View>
      {errors.password ? (
        <Text style={styles.errorText}>{errors.password}</Text>
      ) : (
        <Text style={styles.helperText}>Password must be at least 12 characters long and contain at least one number</Text>
      )}
      <View style={styles.passwordContainer}>
        <TextInput
          style={[styles.input, styles.passwordInput, errors.confirmPassword ? styles.inputError : null]}
          placeholder="Confirm Password"
          value={confirmPassword}
          onChangeText={(text) => {
            setConfirmPassword(text);
            setErrors(prev => ({ ...prev, confirmPassword: '' }));
          }}
          secureTextEntry={!showConfirmPassword}
        />
        <TouchableOpacity
          style={styles.eyeButton}
          onPress={() => setShowConfirmPassword(!showConfirmPassword)}
        >
          <Image
            source={showConfirmPassword ? require('../icons/seePassword.png') : require('../icons/toggleoffpassword.png')}
            style={styles.eyeIcon}
          />
        </TouchableOpacity>
      </View>
      {errors.confirmPassword && (
        <Text style={styles.errorText}>{errors.confirmPassword}</Text>
      )}
      {errors.general && (
        <Text style={styles.errorText}>{errors.general}</Text>
      )}
      <TouchableOpacity style={styles.button} onPress={handleSignup}>
        <Text style={styles.buttonText}>Sign Up</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={() => router.push('/login')}>
        <Text style={styles.linkText}>Already have an account? Login</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  passwordContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15
  },
  passwordInput: {
    flex: 1,
    marginBottom: 0
  },
  eyeButton: {
    position: 'absolute',
    right: 12,
    padding: 8
  },
  eyeButtonText: {
    fontSize: 20
  },
  eyeIcon: {
    width: 24,
    height: 24,
    resizeMode: 'contain'
  },
  inputError: {
    borderColor: '#ff3b30'
  },
  errorText: {
    color: '#ff3b30',
    fontSize: 12,
    marginBottom: 15,
    marginTop: -10,
    paddingHorizontal: 5
  },
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
    backgroundColor: '#fff'
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
    textAlign: 'center'
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
  helperText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 15,
    marginTop: -10,
    paddingHorizontal: 5
  },
  button: {
    backgroundColor: '#007AFF',
    height: 50,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 10
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold'
  },
  linkText: {
    color: '#007AFF',
    textAlign: 'center',
    marginTop: 20
  }
});