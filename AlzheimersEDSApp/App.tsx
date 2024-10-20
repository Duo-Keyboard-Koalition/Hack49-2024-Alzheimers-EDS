/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React, { useEffect, useState } from 'react';
import type {PropsWithChildren} from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  useColorScheme,
  View,
  Button,
  Platform,
  PermissionsAndroid
} from 'react-native';

import {
  Colors,
  DebugInstructions,
  Header,
  LearnMoreLinks,
  ReloadInstructions,
} from 'react-native/Libraries/NewAppScreen';

import PushNotification from 'react-native-push-notification';
import notifee from '@notifee/react-native';

import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs'; // Import react-native-fs

const audioRecorderPlayer = new AudioRecorderPlayer();


function App(): React.JSX.Element {
  const isDarkMode = useColorScheme() === 'dark';

  const backgroundStyle = {
    backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
  };

  const [recording, setRecording] = useState<boolean>(false);
  const [recordingPath, setRecordingPath] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Request Android permissions
  const requestPermissions = async (): Promise<void> => {
    if (Platform.OS === 'android') {
      try {
          const audioPermission = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO);
          if (audioPermission !== PermissionsAndroid.RESULTS.GRANTED) {
              console.warn('Audio permission denied');
              return; // Exit if audio permission is denied
          }
      } catch (err) {
          console.warn(err);
      }
    }
  };

  async function displayNotif() {
    // Request permissions (required for iOS)
    try {
      await notifee.requestPermission()
  
      // Create a channel (required for Android)
      const channelId = await notifee.createChannel({
        id: 'default',
        name: 'Default Channel',
      });
  
      // Display a notification
      await notifee.displayNotification({
        title: '',
        body: 'Main body content of the notification',
        android: {
          channelId,
          // smallIcon: 'ic', // optional, defaults to 'ic_launcher'.
          // pressAction is needed if you want the notification to open the app when pressed
          pressAction: {
            id: 'default',
          },
        },
      });
      console.log('notif created')
    } catch (error) {
      console.log(error);
    }
  }

  // Start recording
  const startRecording = async (): Promise<void> => {
    console.log('attempt start')
    try {
      setRecording(true);
      const path = `${RNFS.DocumentDirectoryPath}/${Date.now()}.mp3`; // Use app's private storage

      const result = await audioRecorderPlayer.startRecorder(path);
      setRecordingPath(result);
      audioRecorderPlayer.addRecordBackListener((e) => {
        console.log('Recording...', e.currentPosition);
      });
    } catch (error) {
      console.warn('Error starting recording:', error);
    }
  };

  // Stop recording
  const stopRecording = async (): Promise<void> => {
    console.log('attempt stop')
    try {
      const result = await audioRecorderPlayer.stopRecorder();
      setRecording(false);
      audioRecorderPlayer.removeRecordBackListener();
      console.log('Recording stopped:', result);
    } catch (error) {
      console.warn('Error stopping recording:', error);
    }
  };

  // Play recorded audio
  const playRecording = async (): Promise<void> => {
    if (!recordingPath) return;

    try {
      setIsPlaying(true);
      await audioRecorderPlayer.startPlayer(recordingPath);
      audioRecorderPlayer.addPlayBackListener((e) => {
        if (e.currentPosition === e.duration) {
          setIsPlaying(false);
          audioRecorderPlayer.stopPlayer();
        }
      });
    } catch (error) {
      console.warn('Error playing recording:', error);
    }
  };

  // Stop playing audio
  const stopPlaying = async (): Promise<void> => {
    try {
      await audioRecorderPlayer.stopPlayer();
      setIsPlaying(false);
      audioRecorderPlayer.removePlayBackListener();
    } catch (error) {
      console.warn('Error stopping playback:', error);
    }
  };

  useEffect(() => {
    // Configure push notification
    // displayNotif();
    requestPermissions();

    console.log('triggered')
  }, []);

  return (
    <View style={{ padding: 20 }}>
      <Button
        title={recording ? 'Stop Recording' : 'Start Recording'}
        onPress={recording ? stopRecording : startRecording}
      />
      {recordingPath && (
        <>
          <Text style={{ marginVertical: 20 }}>Recording saved to: {recordingPath}</Text>
          <Button
            title={isPlaying ? 'Stop Playback' : 'Play Recording'}
            onPress={isPlaying ? stopPlaying : playRecording}
          />
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({

});

export default App;
